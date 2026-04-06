from pathlib import Path
import os
import sys
import argparse
from tqdm import tqdm
import torch
import gc
import json
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import plotly.express as px

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils import get_L_prompt, get_final_logit_prob, get_completion_token_logprobs
from logitlens import compute_logodds_final_layer, get_logodds_gen, get_logodds_disc
from task_registry import get_task
import tasks  # Triggers task registration

def is_hypernym_task(task):
    """Check if task is any hypernym variant (hypernym, hypernym-cars, hypernym-fruit, etc.)"""
    return task == 'hypernym' or task.startswith('hypernym-')

def is_ifeval_task(task):
    """Check if task is any IFEval variant (ifeval, ifeval-<prompt_name>, etc.)"""
    return task == 'ifeval' or task.startswith('ifeval-')

def is_ambigqa_task(task):
    """Check if task is any AmbigQA variant."""
    return task.startswith('ambigqa-')

def is_plausibleqa_task(task):
    """Check if task is any PlausibleQA variant."""
    return task.startswith('plausibleqa-')


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def compute_gpt2_typicality(completions, task, LL):
    """
    Compute GPT-2 unconditional log probability P(completion) for typicality correction.
    
    Args:
        completions: List of completion texts
        task: Task name (e.g., 'hypernym')
        LL: Data items
        
    Returns:
        List of log probabilities, one per completion
    """
    print("\nLoading GPT-2 for typicality correction...")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = get_device()
    gpt2_model = gpt2_model.to(device)
    gpt2_model.eval()
    print(f"  ✓ GPT-2 loaded on {device}")
    
    typicality_scores = []
    
    print("Computing GPT-2 typicality scores...")
    for completion in tqdm(completions, desc="GPT-2 typicality"):
        with torch.no_grad():
            # Tokenize without special tokens
            input_ids = gpt2_tokenizer.encode(completion, add_special_tokens=False)
            
            if len(input_ids) == 0:
                typicality_scores.append(float('-inf'))
                continue
            
            # For single token, compute P(token)
            if len(input_ids) == 1:
                context_ids = gpt2_tokenizer.encode("", add_special_tokens=True)
                full_ids = context_ids + input_ids
                
                input_tensor = torch.tensor([full_ids]).to(device)
                outputs = gpt2_model(input_tensor)
                logits = outputs.logits
                
                target_logits = logits[0, len(context_ids) - 1, :]
                probs = torch.softmax(target_logits, dim=-1)
                token_prob = probs[input_ids[0]].item()
                
                typicality_scores.append(np.log(token_prob + 1e-12))
            else:
                # For multi-token, compute product of conditional probabilities
                log_prob_sum = 0.0
                
                for i in range(len(input_ids)):
                    if i == 0:
                        context_ids = gpt2_tokenizer.encode("", add_special_tokens=True)
                    else:
                        context_ids = gpt2_tokenizer.encode("", add_special_tokens=True)[:-1] + input_ids[:i]
                    
                    max_ctx = 1024
                    if len(context_ids) > max_ctx - 1:
                        context_ids = context_ids[-(max_ctx - 1):]

                    full_ids = context_ids + [input_ids[i]]
                    input_tensor = torch.tensor([full_ids]).to(device)
                    outputs = gpt2_model(input_tensor)
                    logits = outputs.logits
                    
                    target_logits = logits[0, len(context_ids) - 1, :]
                    probs = torch.softmax(target_logits, dim=-1)
                    token_prob = probs[input_ids[i]].item()
                    
                    log_prob_sum += np.log(token_prob + 1e-12)
                
                typicality_scores.append(log_prob_sum)
    
    # Clean up GPT-2 model
    del gpt2_model
    del gpt2_tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"  ✓ Computed {len(typicality_scores)} typicality scores")
    print(f"  Mean typicality: {np.mean(typicality_scores):.4f}")
    print(f"  Std typicality: {np.std(typicality_scores):.4f}")
    
    return typicality_scores


def compute_self_typicality(completions, is_chat=False, has_system_role=False):
    """
    Compute self-typicality: unconditional log P_model(completion) using the scoring model itself.

    Instead of using GPT-2, this computes P(y|null) using the already-loaded model,
    where the null context is BOS (non-chat) or chat-formatted empty prompt (chat models).

    Uses the same is_chat/has_system_role as main scoring so typicality matches
    the model's actual prior distribution.
    """
    typicality_scores = []

    print("\nComputing self-typicality scores (using scoring model itself)...")
    for completion in tqdm(completions, desc="Self typicality"):
        # Use get_completion_token_logprobs with empty prompt; format matches main scoring
        token_logprobs = get_completion_token_logprobs(
            "", completion, model, tokenizer, device,
            is_chat=is_chat, has_system_role=has_system_role
        )
        typicality_scores.append(float(token_logprobs.sum().item()))

    print(f"  Computed {len(typicality_scores)} self-typicality scores")
    print(f"  Mean self-typicality: {np.mean(typicality_scores):.4f}")
    print(f"  Std self-typicality: {np.std(typicality_scores):.4f}")

    return typicality_scores


def make_negated_gen_prompt(item, task, make_prompt, gen_shots='zero'):
    """Construct a negated generator prompt for the LLR typicality correction.

    Returns the negated prompt string and the completion string.
    The negated prompt asks for an *incorrect* completion, so P(completion | neg_prompt)
    captures the model's belief about how likely the completion is as a wrong answer.

    Always uses zero-shot to avoid negating few-shot examples.
    """
    gen_obj = make_prompt(item, style='generator', shots='zero')
    completion = gen_obj.completion

    if is_hypernym_task(task):
        neg_prompt = gen_obj.prompt.replace(" are a kind of", " are NOT a kind of")
        if neg_prompt == gen_obj.prompt:
            raise ValueError(
                f"Negated prompt unchanged for hypernym task. "
                f"Prompt '{gen_obj.prompt[:80]}' doesn't contain ' are a kind of'. "
                f"--neg-typicality may not work with this variation."
            )
    elif is_plausibleqa_task(task) or is_ambigqa_task(task):
        neg_prompt = gen_obj.prompt.replace("Answer the question:", "Give an incorrect answer to the question:")
        neg_prompt = neg_prompt.replace("\nAnswer:", "\nIncorrect answer:")
        if neg_prompt == gen_obj.prompt:
            raise ValueError(
                f"Negated prompt unchanged for QA task. "
                f"Prompt '{gen_obj.prompt[:80]}' doesn't match expected format."
            )
    elif is_ifeval_task(task):
        neg_prompt = "Give a response that does NOT follow these instructions.\n" + gen_obj.prompt
    else:
        raise NotImplementedError(
            f"--neg-typicality is not implemented for task '{task}'. "
            f"Add a negation strategy to make_negated_gen_prompt()."
        )

    return neg_prompt, completion


def compute_neg_typicality(LL, task, make_prompt, gen_shots, is_chat=False, has_system_role=False):
    """Compute log P(completion | negated_prompt) for each item using the scoring model.

    This is the LLR denominator: instead of P(y) (unconditional), we use
    P(y | "give an incorrect answer to Q") which better captures the model's
    belief about what a wrong answer looks like.
    """
    neg_scores = []

    print("\nComputing negated-prompt typicality (LLR denominator)...")
    for item in tqdm(LL, desc="Neg typicality"):
        neg_prompt, completion = make_negated_gen_prompt(item, task, make_prompt, gen_shots)
        token_logprobs = get_completion_token_logprobs(
            neg_prompt, completion, model, tokenizer, device,
            is_chat=is_chat, has_system_role=has_system_role
        )
        neg_scores.append(float(token_logprobs.sum().item()))

    print(f"  Computed {len(neg_scores)} negated-prompt scores")
    print(f"  Mean neg score: {np.mean(neg_scores):.4f}")
    print(f"  Std neg score: {np.std(neg_scores):.4f}")

    return neg_scores


def load_self_vocab_probs(model, tokenizer, is_chat=False, has_system_role=False):
    """
    Compute the scoring model's own unconditional next-token log probabilities
    for all tokens in its vocabulary.

    This is the self-typicality equivalent of load_gpt2_vocab_probs().
    For non-chat models: P_model(token | BOS).
    For chat models: P_model(token | chat-formatted empty prompt), matching main scoring.

    Returns:
        torch.Tensor: log probabilities for each token index
    """
    from utils import get_model_input_device
    model_device = get_model_input_device(model, device)

    print("Computing self-model unconditional vocab probabilities...")
    if is_chat:
        if has_system_role:
            message = [
                {"role": "system", "content": "Answer directly without explanation."},
                {"role": "user", "content": ""},
            ]
        else:
            message = [{"role": "user", "content": ""}]
        prefix_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=False,
        )[0]
        input_ids = prefix_ids.unsqueeze(0)
    else:
        input_ids = tokenizer("", return_tensors="pt")["input_ids"]
        # If tokenizer produces no tokens for empty string (e.g., Qwen — no BOS),
        # use eos/pad token as minimal context for unconditional distribution.
        if input_ids.shape[1] == 0:
            fallback_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
            input_ids = torch.tensor([[fallback_id]])

    with torch.no_grad():
        outputs = model(input_ids.to(model_device), use_cache=False)
        logits = outputs.logits[0, -1, :]  # logits at last position (predicting next token)
        log_probs = torch.log_softmax(logits, dim=-1)

    result = log_probs.cpu().float()
    print(f"  Computed unconditional log probs for {len(result)} tokens")
    print(f"  Mean log prob: {result.mean():.4f}")
    print(f"  Max log prob: {result.max():.4f} (token: {tokenizer.decode([result.argmax().item()])})")

    return result


device = get_device()
yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
no_words = ["No", " No", "NO", "no", " no"]


def init_model(model_name, device, fp32_model=False):
    global model
    global tokenizer
    global terminators 
    # Use float32 if requested to avoid bfloat16 logit quantization
    # (at logit magnitudes ~20, bfloat16 has precision of 0.125, causing
    # log-odds to be quantized to 0.125 steps)
    torch_dtype = torch.float32 if fp32_model else torch.bfloat16
    
    if fp32_model:
        print("Using float32 precision (avoids bfloat16 logit quantization, uses ~2x memory)")
    
    # Common kwargs for loading
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",  # Automatically spread across available GPUs
        "low_cpu_mem_usage": True,
    }
    
    # if 'gemma-3' in model_name:
    #     model = Gemma3ForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    
    if 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            attn_implementation="eager", 
            **load_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    # Note: Don't call model.to(device) when using device_map="auto"
    print("model.config.torch_dtype:", model.config.torch_dtype)
    print(f"Model distributed across devices: {set(model.hf_device_map.values()) if hasattr(model, 'hf_device_map') else 'single device'}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in model_name:
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


def get_base_model_name(modelname):
    modelname = modelname.split("output")[-1]
    return modelname.replace('/', '-')

def get_labels(task, LL):
    """Extract binary labels (1=positive, 0=negative) from data."""
    # Check task registry first (for new extensible tasks)
    task_config = get_task(task)
    if task_config is not None:
        # NEW PATH: Use registered task configuration
        return [1 if task_config['get_label'](i) == 'yes' else 0 for i in LL]
    # LEGACY PATH: Existing task implementations (unchanged)
    if is_hypernym_task(task):
        return [1 if i.taxonomic.strip().capitalize() == 'Yes' else 0 for i in LL]
    elif task=="trivia-qa":
        if 'correct' in LL[0]:
            return [1 if i['correct'] == 'Yes' else 0 for i in LL]
        else:
            return [1 for i in LL]
    elif task=='swords':
        return [1 if i.synonym.capitalize() == 'Yes' else 0 for i in LL]
    elif task=='lambada':
        if 'correct' in LL[0]:
            return [1 if i['correct'] == 'Yes' else 0 for i in LL]
        else:
            return [1 for i in LL]
    elif task=='ifeval':
        return [1 if i['correct'] == 'Yes' else 0 for i in LL]
    elif task=='collie':
        return [1 if i['satisfies_constraint'] else 0 for i in LL]
    else:
        raise ValueError(f"Unknown task: {task}")

def get_example_details(task, LL):
    if is_hypernym_task(task):
        return [ (i.noun1, i.noun2) for i in LL ]
    else:
        raise ValueError(f"Unknown task: {task}")

def create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logodds'):
    """
    Create scatter plot of generator vs validator log-odds or log-probs.
    
    Args:
        logodds_gen: List of generator log-odds/log-probs
        logodds_disc: List of discriminator/validator log-odds/log-probs
        labels: List of binary labels (1=positive, 0=negative)
        modelname: Model name for filename
        task: Task name for filename
        args: Command line arguments
        metric_type: 'logodds' or 'logprobs' for axis labels and filename
    """
    # Convert to numpy arrays
    logodds_gen_np = np.array([float(x) for x in logodds_gen])
    logodds_disc_np = np.array([float(x) for x in logodds_disc])
    labels_np = np.array(labels)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot positive examples (orange)
    pos_mask = labels_np == 1
    plt.scatter(logodds_gen_np[pos_mask], logodds_disc_np[pos_mask], 
                c='orange', label='Positive', alpha=0.6, s=30)
    
    # Plot negative examples (blue)
    neg_mask = labels_np == 0
    plt.scatter(logodds_gen_np[neg_mask], logodds_disc_np[neg_mask],
                c='blue', label='Negative', alpha=0.6, s=30)
    
    # Styling based on metric type
    metric_label = 'log-odds' if metric_type == 'log-odds' else 'log-probs'
    plt.xlabel(f'Generator {metric_label}', fontsize=12)
    plt.ylabel(f'Validator {metric_label}', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Class', fontsize=10, title_fontsize=11)
    
    # Add horizontal line at threshold for validator classification
    if metric_type == 'log-odds':
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='threshold = 0')
    else:
        threshold = np.log(0.5)
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'threshold = log(0.5) ≈ {threshold:.3f}')
    
    # Generate filename with metric type and eval settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Normalize model name for consistent filenames:
    # Base model "google/gemma-2-2b" -> "v6-google_gemma-2-2b"
    # Fine-tuned "../models/v6-google--gemma-2-2b-delta..." -> "v6-google_gemma-2-2b-delta..."
    if '/' in modelname and not modelname.startswith('.'):
        # Base model path like "google/gemma-2-2b" - add v6- prefix for consistency
        model_short = 'v6-' + modelname.replace('/', '_')
    else:
        # Fine-tuned model path like "../models/v6-google--gemma-2-2b-delta..."
        model_short = modelname.split('/')[-1]  # Get last part of path
        # Replace -- with _ (google--gemma -> google_gemma)
        model_short = model_short.replace('--', '_')
    split = "train" if args.train else "test"
    v2_suffix = "_v2" if not args.no_v2 else ""
    eval_tc_suffix = "_tc" if args.typicality_correction else ""
    eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
    self_pfx = "neg-" if getattr(args, 'neg_typicality', False) else ("self-" if getattr(args, 'self_typicality', False) else "")
    filename = f"../outputs/viz_{self_pfx}{model_short}_{task}_{split}_{metric_type}{v2_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.png"

    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")
    plt.close()



def create_visualization_interactive(logodds_gen, logodds_disc, labels, example_details, modelname, task, args, metric_type='logodds'):
    """
    Create an interactive scatter plot (Plotly) of generator vs validator log-odds/log-probs.
    """
    # Convert to numpy
    logodds_gen_np = np.array([float(x) for x in logodds_gen])
    logodds_disc_np = np.array([float(x) for x in logodds_disc])
    labels_np = np.array(labels)
    
    # Prepare dataframe-like dict for Plotly
    data = {
        "Generator": logodds_gen_np,
        "Validator": logodds_disc_np,
        "Label": ["Positive" if l == 1 else "Negative" for l in labels_np],
        "Index": np.arange(len(labels_np)),   # helps identify points
        "Example Details": example_details
    }

    metric_label = "log-odds" if metric_type == "log-odds" else "log-probs"

    # Create interactive scatter
    fig = px.scatter(
        data,
        x="Generator",
        y="Validator",
        color="Label",
        hover_data=["Index", "Generator", "Validator", "Label", "Example Details"],
        color_discrete_map={"Positive": "orange", "Negative": "blue"},
        labels={"Generator": f"Generator {metric_label}",
                "Validator": f"Validator {metric_label}"}
    )

    fig.update_layout(
        title=f"Generator vs Validator ({metric_label})",
        width=900,
        height=700
    )

    # Output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Normalize model name for consistent filenames:
    # Base model "google/gemma-2-2b" -> "v6-google_gemma-2-2b"
    # Fine-tuned "../models/v6-google--gemma-2-2b-delta..." -> "v6-google_gemma-2-2b-delta..."
    if '/' in modelname and not modelname.startswith('.'):
        # Base model path like "google/gemma-2-2b" - add v6- prefix for consistency
        model_short = 'v6-' + modelname.replace('/', '_')
    else:
        # Fine-tuned model path like "../models/v6-google--gemma-2-2b-delta..."
        model_short = modelname.split('/')[-1]  # Get last part of path
        # Replace -- with _ (google--gemma -> google_gemma)
        model_short = model_short.replace('--', '_')
    split = "train" if args.train else "test"
    v2_suffix = "_v2" if not args.no_v2 else ""
    self_pfx = "neg-" if getattr(args, 'neg_typicality', False) else ("self-" if getattr(args, 'self_typicality', False) else "")
    filename = f"../outputs/viz_interactive_{self_pfx}{model_short}_{task}_{split}_{metric_type}{v2_suffix}_{timestamp}.html"

    fig.write_html(filename)
    print(f"Interactive visualization saved to: {filename}")

def load_gpt2_vocab_probs(modelname, tokenizer):
    """
    Load precomputed GPT-2 log probabilities for all tokens in model's vocabulary.
    
    The .npy file structure:
    - 1D array of shape (vocab_size,)
    - Index i contains log P_gpt2(text) where text = tokenizer.decode([i])
    - Created by precompute_gpt2_vocab_probs.py
    
    Args:
        modelname: Model name/path
        tokenizer: The model's tokenizer (used for vocab size verification)
    
    Returns:
        torch.Tensor: log probabilities for each token index, or None if not available
    """
    # Detect vocab size to determine which precomputed file to use
    vocab_size = len(tokenizer)
    
    # Map vocab sizes to their precomputed files
    if vocab_size == 256000:
        # Gemma-2-2b or fine-tuned versions
        vocab_file = Path(__file__).parent.parent / "typicality" / "gpt2_vocab_logprobs.npy"
        expected_vocab_size = 256000
        print(f"  Detected Gemma-2-2b vocabulary (size: {vocab_size})")
    else:
        print(f"Warning: No precomputed GPT-2 vocab probabilities for vocab size {vocab_size}")
        print(f"  Available: Gemma-2-2b (256000 tokens)")
        return None
    
    if not vocab_file.exists():
        print(f"Warning: GPT-2 vocab file not found at {vocab_file}")
        return None
    
    print(f"Loading precomputed GPT-2 vocab probabilities from {vocab_file}...")
    gpt2_log_probs = np.load(vocab_file)
    print(f"  ✓ Loaded {len(gpt2_log_probs)} token probabilities")
    
    # Verify vocab size matches
    actual_vocab_size = len(tokenizer)
    if len(gpt2_log_probs) != actual_vocab_size:
        raise ValueError(
            f"Vocab size mismatch! GPT-2 vocab file has {len(gpt2_log_probs)} entries, "
            f"but model tokenizer has {actual_vocab_size} tokens. "
            f"Expected {expected_vocab_size} for {modelname}."
        )
    
    print(f"  ✓ Vocab size verified: {actual_vocab_size} tokens")
    return torch.tensor(gpt2_log_probs, dtype=torch.float32)

def main(args):
    task = args.task
    modelname = args.model
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots
    print(f"gen_shots: {gen_shots}, disc_shots: {disc_shots}")

    train_flag = args.train
    split_type = args.split_type

    # Default to full completion logprobs (multi-token), unless --no-full-completion-logprobs is set
    use_full_completion_logprobs = not args.no_full_completion_logprobs

    if args.neg_typicality:
        self_prefix = "neg-"
    elif args.self_typicality:
        self_prefix = "self-"
    else:
        self_prefix = ""

    v2 = not args.no_v2
    L_train, L_test, make_prompt = get_L_prompt(task, split_type, seed, sample_negative = args.sample_negative, variation = args.variation, v2=v2)
    print("Loaded data with negative_sample = {}, v2 = {}!".format(args.sample_negative, v2))
    device = get_device()
    print(f"Using device: {device}")

    init_model(modelname, device, fp32_model=args.fp32_model)
    


    # Determine first_sw_token based on whether the tokenizer prepends a BOS token.
    # Gemma and Llama prepend BOS: encode("a X") = [BOS, a, X] -> first_sw_token=2 (base), 1 (chat)
    # Qwen does NOT prepend BOS: encode("a X") = [a, X] -> first_sw_token=1 (base), 0 (chat)
    # NOTE: tokenizer.add_bos_token is unreliable (Gemma-it and Llama-it report False but DO prepend BOS).
    # Always use empirical test.
    test_enc = tokenizer.encode("a test")
    has_bos = (tokenizer.bos_token_id is not None and len(test_enc) > 0 and test_enc[0] == tokenizer.bos_token_id)
    first_sw_token = 2 if has_bos else 1

    model_is_chat = False
    model_has_system_role = False
    if 'instruct' in modelname.lower() or '-it' in modelname.lower():
        model_is_chat = True
        first_sw_token = first_sw_token - 1  # Chat models don't use "a " prefix
        print("Model is chat model!")
    if 'llama' in modelname.lower() or 'qwen' in modelname.lower():
        model_has_system_role = True
        print("Model has system role!")
    if "gpt" in modelname.lower():
        raise ValueError("If you are using GPT then rewrite this bit!")
    print(f"first_sw_token={first_sw_token}, has_bos={has_bos}")

    yestoks = [tokenizer.encode(i)[-1] for i in yes_words]
    notoks = [tokenizer.encode(i)[-1] for i in no_words]

    if train_flag:
        LL = L_train
        train_suffix = "--train"
    else:
        LL = L_test
        train_suffix = ""

    # Filter for single-token completions if requested
    if args.single_token_only:
        print(f"Original dataset size: {len(LL)}")
        filtered_LL = []
        for item in LL:
            gen_prompt_obj = make_prompt(item, style="generator", shots=gen_shots)
            completion_tokens = tokenizer.encode(gen_prompt_obj.completion, add_special_tokens=False)
            if len(completion_tokens) == 1:
                filtered_LL.append(item)
        LL = filtered_LL
        print(f"Filtered to single-token completions: {len(LL)}")
        train_suffix += "--single-token"

    if split_type=='random':
        split_suffix = ""
    elif split_type=='hyper':
        split_suffix = "--hyper"
    elif split_type=='both':
        split_suffix = "--both"
    else:
        raise ValueError()
    
    P_gen = []
    P_disc = []
    gen_sum_logprobs = []
    disc_probs = []

    json_list = []
    # Storage for detailed CSV output
    all_prompts_gen = []
    all_prompts_disc = []
    all_num_tokens = []
    
    # LL = LL[:10]
    for item in tqdm(LL):
        gen_obj = make_prompt(item, style='generator', shots=gen_shots)
        prompt_gen = gen_obj.prompt
        completion_gen = gen_obj.completion
        prompt_disc = make_prompt(item, style='discriminator', shots=disc_shots).prompt
        
        # Store prompts and num_tokens for detailed CSV
        all_prompts_gen.append(prompt_gen)
        all_prompts_disc.append(prompt_disc)
        completion_tokens = tokenizer.encode(completion_gen, add_special_tokens=False)
        all_num_tokens.append(len(completion_tokens))
        probs_gen = get_final_logit_prob(prompt_gen, model, tokenizer, device, is_chat = model_is_chat, has_system_role=model_has_system_role)
        P_gen.append(probs_gen)
        # Compute summed generator log-prob across all completion tokens (conditioned autoregressively)
        if use_full_completion_logprobs:
            gen_token_logprobs = get_completion_token_logprobs(prompt_gen, completion_gen, model, tokenizer, device, is_chat=model_is_chat, has_system_role=model_has_system_role)
            gen_sum_logprobs.append(float(gen_token_logprobs.sum().item()))
        probs_disc = get_final_logit_prob(prompt_disc, model, tokenizer, device, is_chat = model_is_chat, has_system_role=model_has_system_role)
        disc_probs.append((float(probs_disc[yestoks].sum().item()), float(probs_disc[notoks].sum().item())))

        # # DEBUG: Print prompts and probabilities for first 5 examples
        # if len(P_disc) < 5:
        #     # Get Yes/No tokens
        #     if len(P_disc) == 0:
        #         yes_token_strings = ['Yes', ' Yes', 'yes', ' yes']
        #         no_token_strings = ['No', ' No', 'no', ' no']
        #         yestoks = [tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_token_strings]
        #         notoks = [tokenizer.encode(s, add_special_tokens=False)[0] for s in no_token_strings]
        #         print("\n" + "="*80)
        #         print(f"DEBUG: use_full_completion_logprobs = {use_full_completion_logprobs}")
        #         print("="*80)
        #     
        #     p_yes = probs_disc[yestoks].sum()
        #     p_no = probs_disc[notoks].sum()
        #     log_odds = float(torch.log(p_yes / p_no))
        #     log_prob_yes = float(torch.log(p_yes))
        #     
        #     # Get ground truth
        #     if task == 'collie':
        #         gt_label = "POSITIVE" if item['satisfies_constraint'] else "NEGATIVE"
        #     elif task == 'hypernym':
        #         gt_label = "POSITIVE" if item.taxonomic.strip().capitalize() == 'Yes' else "NEGATIVE"
        #     else:
        #         gt_label = "UNKNOWN"
        #     
        #     print(f"\n--- Example {len(P_disc)} (Ground Truth: {gt_label}) ---")
        #     print(f"Discriminator Prompt:\n{prompt_disc}")
        #     print(f"\nProbabilities:")
        #     print(f"  P(Yes)={p_yes:.6f}, P(No)={p_no:.6f}, P(other)={1-p_yes-p_no:.6f}")
        #     print(f"  log-odds={log_odds:.4f}, log-prob={log_prob_yes:.4f}")
        
        P_disc.append(probs_disc)
        if args.train:
            prefix = " " if not model_is_chat else ""
            # Check task registry first (for new extensible tasks)
            task_config = get_task(task)
            if task_config is not None:
                # NEW PATH: Use registered task configuration
                label = task_config['get_label'](item)
                json_list.append({
                    "generator-prompt": prompt_gen,
                    "discriminator-prompt": prompt_disc,
                    "generator-log-prob": 0,
                    "discriminator-log-prob": 0,
                    "generator-completion": prefix + task_config['get_completion'](item).strip(),
                    "discriminator-gold-completion": prefix + ("Yes" if label == 'yes' else "No")
                })
            # LEGACY PATH: Existing task implementations (unchanged)
            elif is_hypernym_task(task):
                json_list.append({"noun1":item.noun1, "noun2":item.noun2, "taxonomic":item.taxonomic, "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0, 
                                "generator-completion": prefix + item.noun2.strip(), "discriminator-gold-completion": prefix + item.taxonomic.strip().capitalize()})
            elif task == 'trivia-qa':
                # print(item.keys())
                if args.sample_negative:
                    json_list.append({"question":item['question'], "answer":prefix + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['answers'][0].strip(), "discriminator-gold-completion": prefix + item['correct']})
                else:
                    json_list.append({"question":item['question'], "answer":prefix + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['answers'][0].strip(), "discriminator-gold-completion": prefix + 'Yes'})
            elif task == 'lambada':
                if args.sample_negative:
                    json_list.append({"context":item['context'], "completion":prefix + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['final_word'].strip(), "discriminator-gold-completion": prefix + item['correct']})
                else:
                    json_list.append({"context":item['context'], "completion":prefix + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['final_word'].strip(), "discriminator-gold-completion": prefix + 'Yes'})
            elif task == 'swords':
                json_list.append({"context":item.context, "target":item.target, "replacement":item.replacement, "synonym":item.synonym,
                                "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": item.replacement.strip(), "discriminator-gold-completion": prefix + item.synonym.strip().capitalize()})
            elif task == "ifeval":
                json_list.append({"prompt":item.prompt, "generator-completion":item.response, "discriminator-gold-completion":item.correct})
            else:
                raise NotImplementedError("Not a task")
            # print(json_list[-1])
    
    # Compute generator scores for typicality correction (if needed)
    # NOTE: disc_scores are computed in compute_logodds_final_layer and returned in res_dict
    # NOTE: When typicality correction is NOT applied, these are recomputed in compute_logodds_final_layer
    if use_full_completion_logprobs:
        # Multi-token case: use log-probs for generator (gen_sum_logprobs is already list of floats)
        gen_scores = gen_sum_logprobs  # No need to wrap in tensor - will be converted to float anyway
        # # OLD: computed disc here, now done in compute_logodds_final_layer
        # logodds_disc = [torch.log(torch.sum(P_disc[ii][..., yestoks], dim=-1)) for ii in range(len(P_disc))]
    else:
        # Single-token case: use log-odds for generator
        gen_scores = [get_logodds_gen(P_gen, LL, ii, tokenizer, first_sw_token, task, is_chat = model_is_chat, use_lgo=True) for ii in range(len(P_gen))]
        # # OLD: computed disc here, now done in compute_logodds_final_layer
        # logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]
    
    # Apply typicality correction if requested
    if args.typicality_correction:
        if args.neg_typicality:
            typ_source = "NEGATED-PROMPT (LLR)"
        elif args.self_typicality:
            typ_source = "SELF-MODEL"
        else:
            typ_source = "GPT-2"
        print("\n" + "="*60)
        print(f"APPLYING TYPICALITY CORRECTION (source: {typ_source})")
        print("="*60)

        if not use_full_completion_logprobs and not args.neg_typicality:
            if args.self_typicality:
                vocab_logprobs = load_self_vocab_probs(
                    model, tokenizer,
                    is_chat=model_is_chat, has_system_role=model_has_system_role
                )
            else:
                vocab_logprobs = load_gpt2_vocab_probs(modelname, tokenizer)

            if vocab_logprobs is not None:
                vocab_logprobs = vocab_logprobs.to(P_gen[0].device)
                print(f"\nApplying PMI correction to probability distributions (prior: {typ_source})...")
                P_gen_corrected = []
                for ii, probs in enumerate(P_gen):
                    log_probs_model = torch.log(probs)
                    log_probs_corrected = log_probs_model - vocab_logprobs
                    P_gen_corrected.append(log_probs_corrected)
            else:
                P_gen_corrected = None
                print("\n  WARNING: No precomputed GPT-2 vocab probs for this tokenizer vocab size.")
                print("  Rank-based metrics (gen_acc, gen_mrr) will use UNCORRECTED distributions.")
                print("  Per-completion gen_score_typcorr will still be computed correctly.")
        else:
            P_gen_corrected = None

        # Compute per-completion typicality scores
        if args.neg_typicality:
            if not use_full_completion_logprobs:
                print("  WARNING: --neg-typicality with --no-full-completion-logprobs: "
                      "neg scores use full completion logprobs while gen scores are single-token. "
                      "This is only correct if completions are single-token (e.g., hypernymy).")
            typicality_scores = compute_neg_typicality(
                LL, task, make_prompt, gen_shots,
                is_chat=model_is_chat, has_system_role=model_has_system_role
            )
        elif args.self_typicality:
            completions = []
            for item in LL:
                gen_obj = make_prompt(item, style='generator', shots=gen_shots)
                completions.append(gen_obj.completion)
            typicality_scores = compute_self_typicality(
                completions,
                is_chat=model_is_chat, has_system_role=model_has_system_role
            )
        else:
            completions = []
            for item in LL:
                gen_obj = make_prompt(item, style='generator', shots=gen_shots)
                completions.append(gen_obj.completion)
            typicality_scores = compute_gpt2_typicality(completions, task, LL)

        # Apply correction to completion scores: corrected_gen = gen - typicality
        if args.neg_typicality:
            correction_label = "log P_model(completion|negated_prompt)"
        elif args.self_typicality:
            correction_label = "log P_model(completion)"
        else:
            correction_label = "log P_GPT2(completion)"
        print(f"\nApplying correction to completion scores: log P(completion|context) - {correction_label}")
        gen_scores_raw = gen_scores.copy() if isinstance(gen_scores, list) else list(gen_scores)
        gen_scores_raw = [float(x) for x in gen_scores_raw]  # Ensure floats
        gen_scores_typcorr = [float(gen_scores[i]) - typicality_scores[i] for i in range(len(gen_scores))]
        gen_scores = gen_scores_typcorr  # Use corrected scores for downstream processing

        print(f"  Original score mean: {np.mean(gen_scores_raw):.4f}")
        print(f"  Corrected score mean ({typ_source}): {np.mean(gen_scores):.4f}")
        print(f"  Correction applied to {len(gen_scores)} examples")
        if not use_full_completion_logprobs and P_gen_corrected is not None:
            print(f"  Full vocab distributions corrected (in log space): {len(P_gen_corrected)} examples")
            # Replace P_gen with corrected version for rank computation
            P_gen = P_gen_corrected

        print("="*60 + "\n")
    else:
        # No typicality correction - set raw scores and leave typcorr as None
        gen_scores_raw = [float(x) for x in gen_scores] if isinstance(gen_scores, list) else [float(x) for x in gen_scores]
        gen_scores_typcorr = None  # Will be NaN in CSV
    
    # # OLD: Confusion matrix was computed here before compute_logodds_final_layer
    # # Now moved to after compute_logodds_final_layer to use consistent disc_scores
    # # Compute confusion matrix for discriminator
    # # Get ground truth labels (1=positive, 0=negative)
    # true_labels = get_labels(task, LL)
    # 
    # # Determine threshold based on metric type
    # # For log-odds: threshold = 0 (since log(P(yes)/P(no)) = 0 when P(yes) = P(no) = 0.5)
    # # Threshold depends on whether we're using log-odds or log-probs for validator
    # # For log-odds: threshold = 0 (since log(P(yes)/P(no)) = 0 when P(yes) = P(no) = 0.5)
    # # For log-probs: threshold = log(0.5) ≈ -0.693 (since log(P(yes)) = log(0.5) when P(yes) = 0.5)
    # if args.validator_log_odds:
    #     threshold = 0.0  # log-odds threshold
    #     metric_name = "log-odds"
    # else:
    #     threshold = np.log(0.5)  # log-probs threshold
    #     metric_name = "log-probs"
    # 
    # # Make predictions: predicted = 1 if score > threshold, else 0
    # disc_scores = np.array([float(x) for x in logodds_disc])
    # pred_labels = (disc_scores > threshold).astype(int)
    # true_labels_np = np.array(true_labels)
    # 
    # # Compute confusion matrix components
    # tp = np.sum((pred_labels == 1) & (true_labels_np == 1))  # True Positives
    # fp = np.sum((pred_labels == 1) & (true_labels_np == 0))  # False Positives
    # tn = np.sum((pred_labels == 0) & (true_labels_np == 0))  # True Negatives
    # fn = np.sum((pred_labels == 0) & (true_labels_np == 1))  # False Negatives
    # 
    # # Print confusion matrix
    # print(f"\nDiscriminator Confusion Matrix ({metric_name}, threshold={threshold:.3f}):")
    # print(f"                 Predicted Positive  Predicted Negative")
    # print(f"Actual Positive        {tp:6d}              {fn:6d}")
    # print(f"Actual Negative        {fp:6d}              {tn:6d}")
    # print(f"\nAccuracy: {(tp + tn) / len(true_labels):.4f}")
    # print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    # print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
    # print(f"F1 Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}\n")

    disc_probs_sum = [p_yes + p_no for p_yes, p_no in disc_probs]
    print(f"Discriminator Probabilities Avg p_yes + p_no: {np.mean(disc_probs_sum):.4f}")
    plt.hist(disc_probs_sum)
    plt.xlabel("Discriminator P(Yes) + P(No)")
    plt.ylabel("Count")
    plt.title(f"Histogram of Discriminator P(Yes) + P(No) for {task}")
    eval_tc_str = "_tc" if args.typicality_correction else ""
    eval_lenorm_str = "_evallenorm" if args.length_normalize else ""
    hist_filename = f"../outputs/hist_disc_probs_{self_prefix}{task}_{modelname.split('/')[-1]}{eval_tc_str}{eval_lenorm_str}.png"
    plt.savefig(hist_filename)
    plt.close()

    # Compute scores via compute_logodds_final_layer (single source of truth for all scores)
    # Pass corrected scores if typicality correction was applied
    corrected_scores = gen_scores if args.typicality_correction else None
    
    res_dict = compute_logodds_final_layer(task,
        P_gen, P_disc, LL, tokenizer, first_sw_token, yestoks, notoks, is_chat=model_is_chat, 
        gen_logprobs=(gen_sum_logprobs if use_full_completion_logprobs else None),
        corrected_logodds_gen=corrected_scores,
        use_log_odds=args.validator_log_odds)

    # Extract scores from res_dict (computed consistently in compute_logodds_final_layer)
    gen_scores = res_dict['gen_scores']
    disc_scores = res_dict['disc_scores']
    disc_threshold = res_dict['disc_threshold']
    
    # Get labels for subsequent uses
    true_labels = get_labels(task, LL)

    # Debug: Save ALL discriminator and generator values with ground truth to CSV
    if args.debug_save_values:
        import csv
        debug_suffix = "logprobs" if use_full_completion_logprobs else "logodds"
        debug_file = f"../outputs/debug_values_{task}_{debug_suffix}.csv"
        
        with open(debug_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header - include task-specific fields for verification
            if is_hypernym_task(task):
                writer.writerow(['index', 'noun1', 'noun2', 'taxonomic', 'ground_truth', 'disc_score', 'gen_score'])
            elif task == 'swords':
                writer.writerow(['index', 'context', 'target', 'replacement', 'synonym', 'ground_truth', 'disc_score', 'gen_score'])
            elif task in ['trivia-qa', 'lambada']:
                writer.writerow(['index', 'ground_truth', 'disc_score', 'gen_score'])
            else:
                writer.writerow(['index', 'ground_truth', 'disc_score', 'gen_score'])
            
            # Write all data
            for i in range(len(disc_scores)):
                if is_hypernym_task(task):
                    writer.writerow([
                        i,
                        LL[i].noun1,
                        LL[i].noun2,
                        LL[i].taxonomic,
                        true_labels[i],
                        float(disc_scores[i]),
                        float(gen_scores[i]) if i < len(gen_scores) else ''
                    ])
                elif task == 'swords':
                    writer.writerow([
                        i,
                        LL[i].context,
                        LL[i].target,
                        LL[i].replacement,
                        LL[i].synonym,
                        true_labels[i],
                        float(disc_scores[i]),
                        float(gen_scores[i]) if i < len(gen_scores) else ''
                    ])
                else:
                    writer.writerow([
                        i,
                        true_labels[i],
                        float(disc_scores[i]),
                        float(gen_scores[i]) if i < len(gen_scores) else ''
                    ])
        print(f"Debug values saved to: {debug_file} ({len(disc_scores)} examples)")
    
    if args.train:
        for jj in range(len(json_list)):
            json_list[jj]["generator-log-prob"] = float(gen_scores[jj])
            json_list[jj]["discriminator-log-prob"] = float(disc_scores[jj])
            # print(json_list[jj])
        print(f"Saving train data to ../data/{task}-train-{modelname.split('/')[-1]}.json")
        with open(f"../data/{task}-train-{modelname.split('/')[-1]}.json", 'w') as f:
            json.dump(json_list, f, indent=4)
        
        # Create visualization if requested
        if args.viz:
            labels = get_labels(task, LL)
            metric_type = 'log-odds' if args.validator_log_odds else 'log-probs'
            create_visualization(gen_scores, disc_scores, labels, modelname, task, args, metric_type=metric_type)
        
        # Save detailed scores to CSV if requested (also works for --train)
        if args.save_scores_csv and is_hypernym_task(task):
            import csv
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Normalize model name for consistent filenames:
            # Base model "google/gemma-2-2b" -> "v6-google_gemma-2-2b"
            # Fine-tuned "../models/v6-google--gemma-2-2b-delta..." -> "v6-google_gemma-2-2b-delta..."
            if '/' in modelname and not modelname.startswith('.'):
                # Base model path like "google/gemma-2-2b" - add v6- prefix for consistency
                model_short = 'v6-' + modelname.replace('/', '_')
            else:
                # Fine-tuned model path like "../models/v6-google--gemma-2-2b-delta..."
                model_short = modelname.split('/')[-1]  # Get last part of path
                # Replace -- with _ (google--gemma -> google_gemma)
                model_short = model_short.replace('--', '_')
            split = "train"
            v2_suffix = "_v2" if not args.no_v2 else ""
            metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
            eval_tc_suffix = "_tc" if args.typicality_correction else ""
            eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
            scores_csv_filename = f"../outputs/scores_{self_prefix}{model_short}_{task}_{split}{v2_suffix}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv"
            
            strategy = f"gen:{gen_shots}_disc:{disc_shots}"
            if use_full_completion_logprobs:
                strategy += "_fullcomp"
            else:
                strategy += "_singletoken"
            if args.validator_log_odds:
                strategy += "_logodds"
            else:
                strategy += "_logprobs"
            if args.typicality_correction:
                if args.neg_typicality:
                    strategy += "_negtypcorr"
                elif args.self_typicality:
                    strategy += "_selftypcorr"
                else:
                    strategy += "_typcorr"
            
            with open(scores_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['noun1', 'noun2', 'num_tokens', 'strategy', 'gpt4_ground_truth', 
                               'gen_score', 'gen_score_typcorr', 'gen_score_lenorm', 'gen_score_typcorr_lenorm',
                               'val_score', 'gen_prompt', 'val_prompt'])
                
                labels = get_labels(task, LL)
                for i, item in enumerate(LL):
                    gen_prompt_final = all_prompts_gen[i].split("\n")[-1]
                    disc_prompt_final = all_prompts_disc[i].split("\n")[-1]
                    item_strategy = getattr(item, 'strategy', strategy)
                    
                    num_toks = all_num_tokens[i]
                    gen_score_raw = gen_scores_raw[i]
                    gen_score_typcorr_val = gen_scores_typcorr[i] if gen_scores_typcorr is not None else float('nan')
                    gen_score_lenorm = gen_score_raw / num_toks if num_toks > 0 else float('nan')
                    gen_score_typcorr_lenorm = gen_score_typcorr_val / num_toks if (gen_scores_typcorr is not None and num_toks > 0) else float('nan')
                    
                    writer.writerow([
                        item.noun1,
                        getattr(item, 'noun2', getattr(item, 'fixed_hypernym_generator', '')),
                        num_toks,
                        item_strategy,
                        item.taxonomic,
                        float(gen_score_raw),
                        float(gen_score_typcorr_val) if gen_scores_typcorr is not None else '',
                        float(gen_score_lenorm),
                        float(gen_score_typcorr_lenorm) if gen_scores_typcorr is not None else '',
                        float(disc_scores[i]),
                        gen_prompt_final,
                        disc_prompt_final
                    ])
            print(f"Detailed scores saved to: {scores_csv_filename}")
        
        elif args.save_scores_csv and is_ifeval_task(task):
            import csv
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = modelname.split('/')[-1].replace('--', '_')
            split = "train"
            metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
            eval_tc_suffix = "_tc" if args.typicality_correction else ""
            eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
            scores_csv_filename = f"../outputs/scores_{self_prefix}{model_short}_{task}_{split}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv"

            def _get_field(obj, key, default=""):
                if hasattr(obj, key):
                    return getattr(obj, key)
                try:
                    return obj[key]
                except Exception:
                    return default

            with open(scores_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'prompt',
                    'response',
                    'num_tokens',
                    'correct',
                    'val_prompt',
                    'val_score',
                    'gen_score',
                    'gen_score_typcorr',
                    'gen_score_lenorm',
                    'gen_score_typcorr_lenorm',
                ])

                for i, item in enumerate(LL):
                    prompt = _get_field(item, 'prompt', '')
                    response = _get_field(item, 'response', _get_field(item, 'generator-completion', ''))
                    correct = _get_field(item, 'correct', _get_field(item, 'discriminator-gold-completion', ''))

                    val_prompt = all_prompts_disc[i]

                    num_toks = all_num_tokens[i]
                    gen_score_raw = gen_scores_raw[i]
                    gen_score_typcorr_val = gen_scores_typcorr[i] if gen_scores_typcorr is not None else float('nan')
                    gen_score_lenorm = gen_score_raw / num_toks if num_toks > 0 else float('nan')
                    gen_score_typcorr_lenorm = gen_score_typcorr_val / num_toks if (gen_scores_typcorr is not None and num_toks > 0) else float('nan')

                    if isinstance(correct, bool):
                        correct = "Yes" if correct else "No"

                    writer.writerow([
                        prompt,
                        response,
                        num_toks,
                        correct,
                        val_prompt,
                        disc_scores[i],
                        gen_score_raw,
                        gen_score_typcorr_val,
                        gen_score_lenorm,
                        gen_score_typcorr_lenorm,
                    ])

            print(f"Detailed scores saved to: {scores_csv_filename}")

        elif args.save_scores_csv and (is_ambigqa_task(task) or is_plausibleqa_task(task)):
            import csv
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if '/' in modelname and not modelname.startswith('.'):
                model_short = 'v6-' + modelname.replace('/', '_')
            else:
                model_short = modelname.split('/')[-1].replace('--', '_')
            split = "train"
            metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
            eval_tc_suffix = "_tc" if args.typicality_correction else ""
            eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
            scores_csv_filename = f"../outputs/scores_{self_prefix}{model_short}_{task}_{split}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv"

            def _get_field(obj, key, default=""):
                if hasattr(obj, key):
                    return getattr(obj, key)
                try:
                    return obj[key]
                except Exception:
                    return default

            with open(scores_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'question', 'answer', 'num_tokens', 'strategy',
                    'gpt4_ground_truth', 'val_prompt', 'val_score',
                    'gen_score', 'gen_score_typcorr', 'gen_score_lenorm',
                    'gen_score_typcorr_lenorm',
                ])

                for i, item in enumerate(LL):
                    question = _get_field(item, 'question', '')
                    answer = _get_field(item, 'answer', '')
                    item_strategy = _get_field(item, 'strategy', '')
                    correct = _get_field(item, 'correct', '').strip().lower()
                    correct_label = 'yes' if correct in ('yes', 'true', '1') else 'no'

                    val_prompt = all_prompts_disc[i].split("\n")[-1]
                    num_toks = all_num_tokens[i]
                    gen_score_raw = gen_scores_raw[i]
                    gen_score_typcorr_val = gen_scores_typcorr[i] if gen_scores_typcorr is not None else float('nan')
                    gen_score_lenorm = gen_score_raw / num_toks if num_toks > 0 else float('nan')
                    gen_score_typcorr_lenorm = gen_score_typcorr_val / num_toks if (gen_scores_typcorr is not None and num_toks > 0) else float('nan')

                    writer.writerow([
                        question, answer, num_toks, item_strategy,
                        correct_label, val_prompt, disc_scores[i],
                        gen_score_raw, gen_score_typcorr_val,
                        gen_score_lenorm, gen_score_typcorr_lenorm,
                    ])

            print(f"Detailed scores saved to: {scores_csv_filename}")

        return

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # # OLD: Duplicate call to compute_logodds_final_layer - now called earlier (before debug section)
    # # Pass corrected scores if typicality correction was applied
    # corrected_scores = gen_scores if args.typicality_correction else None
    # 
    # res_dict = compute_logodds_final_layer(task,
    #     P_gen, P_disc, LL, tokenizer, first_sw_token, yestoks, notoks, is_chat=model_is_chat, 
    #     gen_logprobs=(gen_sum_logprobs if use_full_completion_logprobs else None),
    #     corrected_logodds_gen=corrected_scores,
    #     use_log_odds=args.validator_log_odds)
    #
    # # Extract scores from res_dict (computed consistently in compute_logodds_final_layer)
    # gen_scores = res_dict['gen_scores']
    # disc_scores = res_dict['disc_scores']
    # disc_threshold = res_dict['disc_threshold']
    
    # Compute confusion matrix for discriminator using consistent disc_scores
    metric_name = "log-odds" if args.validator_log_odds else "log-probs"
    
    # Make predictions: predicted = 1 if score > threshold, else 0
    disc_scores_np = np.array([float(x) for x in disc_scores])
    pred_labels = (disc_scores_np > disc_threshold).astype(int)
    true_labels_np = np.array(true_labels)
    
    # Compute confusion matrix components
    tp = np.sum((pred_labels == 1) & (true_labels_np == 1))  # True Positives
    fp = np.sum((pred_labels == 1) & (true_labels_np == 0))  # False Positives
    tn = np.sum((pred_labels == 0) & (true_labels_np == 0))  # True Negatives
    fn = np.sum((pred_labels == 0) & (true_labels_np == 1))  # False Negatives
    
    # Print confusion matrix
    print(f"\nDiscriminator Confusion Matrix ({metric_name}, threshold={disc_threshold:.3f}):")
    print(f"                 Predicted Positive  Predicted Negative")
    print(f"Actual Positive        {tp:6d}              {fn:6d}")
    print(f"Actual Negative        {fp:6d}              {tn:6d}")
    print(f"\nAccuracy: {(tp + tn) / len(true_labels):.4f}")
    print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
    print(f"F1 Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}\n")
    
    basename = get_base_model_name(modelname)


    summary_file = os.path.join("../outputs/eval_results.csv")
    with open(summary_file, 'a') as f:
        # if file is empty:
        if os.stat(summary_file).st_size == 0:
            f.write("model,task,typcorr,corr_all,corr_pos,corr_neg,disc_acc,disc_roc,gen_acc_5,gen_acc_10,gen_acc_40,gen_acc_100,gen_acc_1000,gen_mrr_pos,gen_mrr_neg,gen_shots,disc_shots,split,split_type,seed,spear_all,spear_pos,spear_neg,gen_acc_5_dataset,gen_acc_10_dataset,gen_acc_40_dataset,gen_acc_100_dataset,gen_acc_1000_dataset,gen_mrr_pos_dataset,gen_mrr_neg_dataset,gen_roc,\n")
        split = "train" if args.train else "test"
        typcorr = "Y" if args.typicality_correction else "N"
        f.write(f"{modelname},{task},{typcorr},{res_dict['corr_all']},{res_dict['corr_pos']},{res_dict['corr_neg']},{res_dict['disc_acc']},{res_dict['disc_roc']},{res_dict['gen_acc_dict'][5]},{res_dict['gen_acc_dict'][10]},{res_dict['gen_acc_dict'][40]},{res_dict['gen_acc_dict'][100]},{res_dict['gen_acc_dict'][1000]},{res_dict['gen_mrr_pos']},{res_dict['gen_mrr_neg']},{gen_shots},{disc_shots},{split},{split_type},{seed}")
        f.write(f",{res_dict['spear_all']},{res_dict['spear_pos']},{res_dict['spear_neg']}")
        # Add dataset-constrained metrics
        if 'gen_acc_dict_dataset' in res_dict:
            f.write(f",{res_dict['gen_acc_dict_dataset'][5]},{res_dict['gen_acc_dict_dataset'][10]},{res_dict['gen_acc_dict_dataset'][40]},{res_dict['gen_acc_dict_dataset'][100]},{res_dict['gen_acc_dict_dataset'][1000]},{res_dict['gen_mrr_pos_dataset']},{res_dict['gen_mrr_neg_dataset']}")
        else:
            f.write(",,,,,,,")  # Empty values if not computed
        # Add gen_roc at the end
        f.write(f",{res_dict['gen_roc']},\n")
    
    # Create visualization if requested
    if args.viz:
        labels = get_labels(task, LL)
        metric_type = 'log-odds' if args.validator_log_odds else 'log-probs'
        create_visualization(gen_scores, disc_scores, labels, modelname, task, args, metric_type=metric_type)
        # # OLD: different visualizations for single-token vs multi-token
        # if use_full_completion_logprobs:
        #     # Multi-token: one plot with log-probs
        #     create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logprobs')
        #     # create_visualization_interactive(logodds_gen, logodds_disc, labels, example_details, modelname, task, args, metric_type='logprobs')
        # else:
        #     # Single-token: two plots (log-odds and log-probs)
        #     create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logodds')
        #     create_visualization(logprobs_gen, logprobs_disc, labels, modelname, task, args, metric_type='logprobs')

    # Save detailed scores to CSV if requested
    if args.save_scores_csv and is_hypernym_task(task):
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Normalize model name for consistent filenames:
        # Base model "google/gemma-2-2b" -> "v6-google_gemma-2-2b"
        # Fine-tuned "../models/v6-google--gemma-2-2b-delta..." -> "v6-google_gemma-2-2b-delta..."
        if '/' in modelname and not modelname.startswith('.'):
            # Base model path like "google/gemma-2-2b" - add v6- prefix for consistency
            model_short = 'v6-' + modelname.replace('/', '_')
        else:
            # Fine-tuned model path like "../models/v6-google--gemma-2-2b-delta..."
            model_short = modelname.split('/')[-1]  # Get last part of path
            # Replace -- with _ (google--gemma -> google_gemma)
            model_short = model_short.replace('--', '_')
        split = "train" if args.train else "test"
        v2_suffix = "_v2" if not args.no_v2 else ""
        metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
        # Add eval setting suffixes
        eval_tc_suffix = "_tc" if args.typicality_correction else ""
        eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
        scores_csv_filename = f"../outputs/scores_{self_prefix}{model_short}_{task}_{split}{v2_suffix}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv"
        
        # Determine strategy string
        strategy = f"gen:{gen_shots}_disc:{disc_shots}"
        if use_full_completion_logprobs:
            strategy += "_fullcomp"
        else:
            strategy += "_singletoken"
        if args.validator_log_odds:
            strategy += "_logodds"
        else:
            strategy += "_logprobs"
        if args.typicality_correction:
            if args.neg_typicality:
                strategy += "_negtypcorr"
            elif args.self_typicality:
                strategy += "_selftypcorr"
            else:
                strategy += "_typcorr"
        
        with open(scores_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['noun1', 'noun2', 'num_tokens', 'strategy', 'gpt4_ground_truth', 
                           'gen_score', 'gen_score_typcorr', 'gen_score_lenorm', 'gen_score_typcorr_lenorm',
                           'val_score', 'gen_prompt', 'val_prompt'])
            
            for i, item in enumerate(LL):
                # Extract just the final query part (not the few-shot examples)
                gen_prompt_final = all_prompts_gen[i].split("\n")[-1]
                disc_prompt_final = all_prompts_disc[i].split("\n")[-1]
                # Use strategy from data item if available, otherwise use constructed strategy
                item_strategy = getattr(item, 'strategy', strategy)
                
                # Compute length-normalized scores
                num_toks = all_num_tokens[i]
                gen_score_raw = gen_scores_raw[i]
                gen_score_typcorr_val = gen_scores_typcorr[i] if gen_scores_typcorr is not None else float('nan')
                gen_score_lenorm = gen_score_raw / num_toks if num_toks > 0 else float('nan')
                gen_score_typcorr_lenorm = gen_score_typcorr_val / num_toks if (gen_scores_typcorr is not None and num_toks > 0) else float('nan')
                
                writer.writerow([
                    item.noun1,
                    item.noun2,
                    num_toks,
                    item_strategy,
                    item.taxonomic,
                    gen_score_raw,
                    gen_score_typcorr_val,
                    gen_score_lenorm,
                    gen_score_typcorr_lenorm,
                    disc_scores[i],
                    gen_prompt_final,
                    disc_prompt_final
                ])
        
        print(f"Detailed scores saved to: {scores_csv_filename}")

    elif args.save_scores_csv and is_ifeval_task(task):
        import csv
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = modelname.split('/')[-1].replace('--', '_')
        split = "train" if args.train else "test"
        metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
        # Add eval setting suffixes
        eval_tc_suffix = "_tc" if args.typicality_correction else ""
        eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
        scores_csv_filename = f"../outputs/scores_{self_prefix}{model_short}_{task}_{split}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv"

        # Determine strategy string (kept for debugging/repro; not a required column for IFEval)
        strategy = f"gen:{gen_shots}_disc:{disc_shots}"
        if use_full_completion_logprobs:
            strategy += "_fullcomp"
        else:
            strategy += "_singletoken"
        strategy += "_logodds" if args.validator_log_odds else "_logprobs"
        if args.typicality_correction:
            if args.neg_typicality:
                strategy += "_negtypcorr"
            elif args.self_typicality:
                strategy += "_selftypcorr"
            else:
                strategy += "_typcorr"

        def _get_field(obj, key, default=""):
            if hasattr(obj, key):
                return getattr(obj, key)
            try:
                return obj[key]
            except Exception:
                return default

        with open(scores_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'prompt',
                'response',
                'num_tokens',
                'correct',
                'val_prompt',
                'val_score',
                'gen_score',
                'gen_score_typcorr',
                'gen_score_lenorm',
                'gen_score_typcorr_lenorm',
            ])

            for i, item in enumerate(LL):
                prompt = _get_field(item, 'prompt', '')
                response = _get_field(item, 'response', _get_field(item, 'generator-completion', ''))
                correct = _get_field(item, 'correct', _get_field(item, 'discriminator-gold-completion', ''))

                # Prompts: keep full validator prompt (contains both prompt+response); generator prompt may
                # include few-shot examples so we do not write it unless requested.
                val_prompt = all_prompts_disc[i]

                # Compute length-normalized scores
                num_toks = all_num_tokens[i]
                gen_score_raw = gen_scores_raw[i]
                gen_score_typcorr_val = gen_scores_typcorr[i] if gen_scores_typcorr is not None else float('nan')
                gen_score_lenorm = gen_score_raw / num_toks if num_toks > 0 else float('nan')
                gen_score_typcorr_lenorm = gen_score_typcorr_val / num_toks if (gen_scores_typcorr is not None and num_toks > 0) else float('nan')

                # Normalize correct to Yes/No if it's a bool
                if isinstance(correct, bool):
                    correct = "Yes" if correct else "No"

                writer.writerow([
                    prompt,
                    response,
                    num_toks,
                    correct,
                    val_prompt,
                    disc_scores[i],
                    gen_score_raw,
                    gen_score_typcorr_val,
                    gen_score_lenorm,
                    gen_score_typcorr_lenorm,
                ])

        print(f"Detailed scores saved to: {scores_csv_filename}")

    elif args.save_scores_csv and (is_ambigqa_task(task) or is_plausibleqa_task(task)):
        import csv
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if '/' in modelname and not modelname.startswith('.'):
            model_short = 'v6-' + modelname.replace('/', '_')
        else:
            model_short = modelname.split('/')[-1].replace('--', '_')
        split = "train" if args.train else "test"
        metric_suffix = "_log-odds" if args.validator_log_odds else "_log-probs"
        eval_tc_suffix = "_tc" if args.typicality_correction else ""
        eval_lenorm_suffix = "_evallenorm" if args.length_normalize else ""
        scores_csv_filename = f"../outputs/scores_{self_prefix}{model_short}_{task}_{split}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv"

        def _get_field(obj, key, default=""):
            if hasattr(obj, key):
                return getattr(obj, key)
            try:
                return obj[key]
            except Exception:
                return default

        with open(scores_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'question',
                'answer',
                'num_tokens',
                'strategy',
                'gpt4_ground_truth',
                'val_prompt',
                'val_score',
                'gen_score',
                'gen_score_typcorr',
                'gen_score_lenorm',
                'gen_score_typcorr_lenorm',
            ])

            for i, item in enumerate(LL):
                question = _get_field(item, 'question', '')
                answer = _get_field(item, 'answer', '')
                item_strategy = _get_field(item, 'strategy', '')
                correct = _get_field(item, 'correct', '').strip().lower()
                correct_label = 'yes' if correct in ('yes', 'true', '1') else 'no'

                val_prompt = all_prompts_disc[i].split("\n")[-1]

                num_toks = all_num_tokens[i]
                gen_score_raw = gen_scores_raw[i]
                gen_score_typcorr_val = gen_scores_typcorr[i] if gen_scores_typcorr is not None else float('nan')
                gen_score_lenorm = gen_score_raw / num_toks if num_toks > 0 else float('nan')
                gen_score_typcorr_lenorm = gen_score_typcorr_val / num_toks if (gen_scores_typcorr is not None and num_toks > 0) else float('nan')

                writer.writerow([
                    question,
                    answer,
                    num_toks,
                    item_strategy,
                    correct_label,
                    val_prompt,
                    disc_scores[i],
                    gen_score_raw,
                    gen_score_typcorr_val,
                    gen_score_lenorm,
                    gen_score_typcorr_lenorm,
                ])

        print(f"Detailed scores saved to: {scores_csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute log-odds on test data")
    parser.add_argument("--model", type=str, help="model directory to process (this should contain merged/ subdirectory) or hf model")
    # parser.add_argument("--tunedlens", action="store_true", default=False, help="")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling the data.")
    parser.add_argument("--disc-shots", type=str, default='few', help="'zero' vs 'few'")
    parser.add_argument("--gen-shots", type=str, default='zero', help="'zero' vs 'few'")
    parser.add_argument("--train", action="store_true", default=False, help="log-odds of train or test set?")
    parser.add_argument("--split_type", type=str, default='random', help="'random' vs 'hyper' vs 'both' ")
    parser.add_argument("--task", type=str, default='hypernym', help="hypernym, trivia-qa, etc")
    parser.add_argument("--sample_negative", action="store_true", default=False, help="whether to sample negative examples when loading trivia-qa or lambada")
    parser.add_argument("--variation", type=str, default="0", help="variation parameter for hypernym prompt formatting (default: '0')")
    parser.add_argument("--single_token_only", action="store_true", default=False, help="only use test data where generator completion is exactly one token")
    parser.add_argument("--no-full-completion-logprobs", action="store_true", default=False, help="use single-token log-probs instead of full completion log-probs for generator scoring")
    parser.add_argument("--viz", action="store_true", default=False, help="create and save visualization plot of generator vs validator log-odds")
    parser.add_argument("--debug_save_values", action="store_true", default=False, help="save discriminator log-odds/log-probs values to file for debugging")
    parser.add_argument("--typicality-correction", action="store_true", default=False, help="apply typicality correction using PMI: corrects both completion scores and full vocab distributions for ranking")
    parser.add_argument("--self-typicality", action="store_true", default=False, help="use the scoring model itself for typicality correction instead of GPT-2. Implies --typicality-correction.")
    parser.add_argument("--neg-typicality", action="store_true", default=False, help="use negated prompts for typicality correction (LLR: log P(y|Q) - log P(y|neg_Q)). Implies --typicality-correction.")
    parser.add_argument("--validator-log-odds", action="store_true", default=False, help="use log-odds (log(P(Yes)/P(No))) for validator instead of log-probs (log(P(Yes))). Changes threshold from log(0.5) to 0.")
    parser.add_argument("--no-v2", action="store_true", default=False, help="use original hypernym data instead of v2 grammar-corrected data")
    parser.add_argument("--save-scores-csv", action="store_true", default=False, help="save detailed scores to CSV with all score columns")
    parser.add_argument("--length-normalize", action="store_true", default=False, help="also compute length-normalized gen scores (gen_score / num_tokens)")
    parser.add_argument("--fp32-model", action="store_true", default=False, help="load model in float32 instead of bfloat16 to avoid logit quantization (uses ~2x memory but gives continuous log-odds)")

    args = parser.parse_args()
    if args.neg_typicality and args.self_typicality:
        parser.error("--neg-typicality and --self-typicality are mutually exclusive")
    if args.self_typicality:
        args.typicality_correction = True
    if args.neg_typicality:
        args.typicality_correction = True
    main(args)
