"""
This script is used to train a model to rank discriminator prompts to match the ranking of log-probabilities of generator prompts.

Usage:
python ranking_loss_ref.py --model google/gemma-2-2b --task hypernym --with_ref --num_epochs 10 --learning_rate 1e-5 --delta 5 --total_samples 5110 --save_steps 1

"""
import os
import sys
import itertools
import csv
from collections import defaultdict
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from peft import LoraConfig, get_peft_model
import math
import random
import argparse
import wandb

from datasets import load_dataset
from sklearn.metrics import roc_curve

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
import utils
from utils import make_prompt_triviaqa, make_prompt_hypernymy, make_prompt_swords, make_prompt_lambada, make_prompt_ifeval, make_prompt_collie, get_final_logit_prob, get_completion_token_logprobs
from task_registry import get_task, get_all_task_names

def compute_optimal_threshold(scores, labels):
    """
    Compute the threshold that maximizes accuracy for binary classification.
    
    Args:
        scores: list/array of scores (higher = more likely positive)
        labels: list/array of binary labels (1=positive, 0=negative)
    
    Returns:
        optimal_threshold: the threshold that maximizes accuracy
        best_accuracy: the accuracy achieved at the optimal threshold
    """
    import numpy as np
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Get unique thresholds from ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Compute accuracy at each threshold
    # accuracy = (TP + TN) / N = tpr * P(pos) + (1-fpr) * P(neg)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    accuracies = (tpr * n_pos + (1 - fpr) * n_neg) / len(labels)
    
    # Find threshold with maximum accuracy
    best_idx = np.argmax(accuracies)
    optimal_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    return optimal_threshold, best_accuracy

# good_pair, alpha_fun_1, get_alpha are used for "both" mode
def good_pair(log_prob_i, log_prob_j, label_i, label_j):
    """Determine if a pair is good based on log probabilities and labels."""
    if log_prob_i > log_prob_j and label_i == 1 and label_j == 0:
        return True
    elif log_prob_i <= log_prob_j and label_i == 0 and label_j == 1:
        return True
    else:
        return False

def alpha_fun_1(gen_log_prob_i, gen_log_prob_j, disc_log_prob_i, disc_log_prob_j, label_i, label_j):
    """Determine alpha based on which model has the good pair."""
    if good_pair(gen_log_prob_i, gen_log_prob_j, label_i, label_j) and not good_pair(disc_log_prob_i, disc_log_prob_j, label_i, label_j):
        return 1.0  # g->v direction
    elif good_pair(disc_log_prob_i, disc_log_prob_j, label_i, label_j) and not good_pair(gen_log_prob_i, gen_log_prob_j, label_i, label_j):
        return 0.0  # v->g direction
    else:
        return 0.5  # equal weighting

def get_alpha(alpha_arg, gen_log_prob_i, gen_log_prob_j, disc_log_prob_i, disc_log_prob_j, label_i, label_j):
    """Get alpha value based on argument and sample characteristics."""
    if isinstance(alpha_arg, (int, float)):
        return float(alpha_arg)
    elif alpha_arg == "alpha_fun_1":
        return alpha_fun_1(gen_log_prob_i, gen_log_prob_j, disc_log_prob_i, disc_log_prob_j, label_i, label_j)
    else:
        raise ValueError(f"Unknown alpha function: {alpha_arg}")

def compute_gpt2_typicality(completions, tokenizer_gpt2, model_gpt2, device):
    """
    Compute GPT-2 unconditional log probability P(completion) for typicality correction.
    
    Args:
        completions: List of completion texts
        tokenizer_gpt2: GPT-2 tokenizer
        model_gpt2: GPT-2 model
        device: Device to run on
        
    Returns:
        List of log probabilities, one per completion
    """
    import numpy as np
    
    typicality_scores = []
    
    print("Computing GPT-2 typicality scores...")
    for completion in tqdm(completions, desc="GPT-2 typicality"):
        with torch.no_grad():
            # Tokenize without special tokens
            input_ids = tokenizer_gpt2.encode(completion, add_special_tokens=False)
            
            if len(input_ids) == 0:
                typicality_scores.append(float('-inf'))
                continue
            
            # For single token, compute P(token)
            if len(input_ids) == 1:
                context_ids = tokenizer_gpt2.encode("", add_special_tokens=True)
                full_ids = context_ids + input_ids
                
                input_tensor = torch.tensor([full_ids]).to(device)
                outputs = model_gpt2(input_tensor)
                logits = outputs.logits
                
                target_logits = logits[0, len(context_ids) - 1, :]
                probs = torch.softmax(target_logits, dim=-1)
                token_prob = probs[input_ids[0]].item()
                
                typicality_scores.append(math.log(token_prob + 1e-12))
            else:
                # For multi-token, compute product of conditional probabilities
                log_prob_sum = 0.0
                
                for i in range(len(input_ids)):
                    if i == 0:
                        context_ids = tokenizer_gpt2.encode("", add_special_tokens=True)
                    else:
                        context_ids = tokenizer_gpt2.encode("", add_special_tokens=True)[:-1] + input_ids[:i]
                    
                    # Assert: context_ids for i > 0 should also start with BOS like i == 0
                    # If this fails, the [:-1] slice is removing the BOS token incorrectly
                    bos_tokens = tokenizer_gpt2.encode("", add_special_tokens=True)
                    if i == 0:
                        assert context_ids == bos_tokens, f"i=0 context should be BOS: {context_ids} vs {bos_tokens}"
                    else:
                        # Check if context starts with BOS (it should for consistency)
                        expected_context = bos_tokens + input_ids[:i]
                        assert context_ids == expected_context, (
                            f"Context mismatch at i={i}: got {context_ids}, expected {expected_context}. "
                            f"The [:-1] slice removes BOS, making contexts inconsistent between i=0 and i>0."
                        )
                    
                    max_ctx = 1024
                    if len(context_ids) > max_ctx - 1:
                        context_ids = context_ids[-(max_ctx - 1):]
                    
                    full_ids = context_ids + [input_ids[i]]
                    input_tensor = torch.tensor([full_ids]).to(device)
                    outputs = model_gpt2(input_tensor)
                    logits = outputs.logits
                    
                    target_logits = logits[0, len(context_ids) - 1, :]
                    probs = torch.softmax(target_logits, dim=-1)
                    token_prob = probs[input_ids[i]].item()
                    
                    log_prob_sum += math.log(token_prob + 1e-12)
                
                typicality_scores.append(log_prob_sum)
    
    print(f"  ✓ Computed {len(typicality_scores)} typicality scores")
    if len(typicality_scores) > 0:
        print(f"  Mean typicality: {sum(typicality_scores)/len(typicality_scores):.4f}")
    
    return typicality_scores


def track_all_scores(model, tokenizer, L_train_all, task, device, yestoks, notoks, 
                     length_normalize=False, use_full_completion=True, task_config=None,
                     validator_log_odds=True, is_chat=False, has_system_role=False,
                     batch_size=16):
    """
    Compute generator and validator scores for all datapoints in L_train_all.
    
    BATCHED VERSION for speed - processes multiple items per forward pass.
    
    Args:
        validator_log_odds: If True, return log(P(Yes)/P(No)). If False, return log(P(Yes)).
        is_chat: Whether to use chat template for prompts.
        has_system_role: Whether the model supports system role in chat template.
        batch_size: Number of items to process per batch.
    
    Returns a list of dicts with keys: noun1, noun2, gen_score, val_score
    """
    model.eval()
    
    # Determine how to get noun1, noun2, completion based on task
    def get_noun1(item):
        if hasattr(item, 'noun1'):
            return item.noun1
        elif task_config and 'get_noun1' in task_config:
            return task_config['get_noun1'](item)
        return str(item)[:50]  # fallback
    
    def get_noun2(item):
        if hasattr(item, 'noun2'):
            return item.noun2
        elif task_config and 'get_completion' in task_config:
            return task_config['get_completion'](item)
        return ""
    
    def get_item_completion(item):
        """Get the generator completion - must match what make_prompt returns for consistency."""
        if task_config:
            result = task_config['make_prompt'](item, style='generator')
            if hasattr(result, 'completion'):
                return result.completion.lstrip()
        if hasattr(item, 'fixed_hypernym_generator'):
            return item.fixed_hypernym_generator
        elif hasattr(item, 'noun2'):
            return item.noun2
        return ""
    
    def get_val_prompt(item):
        """Get the validator/discriminator prompt."""
        if task_config:
            result = task_config['make_prompt'](item, style='discriminator')
            return result.prompt if hasattr(result, 'prompt') else str(item)
        elif task in ['hypernym', 'hypernym-car']:
            few_shot_prefix = (
                "Do you think bees are furniture? Answer: No\n\n"
                "Do you think corgis are dogs? Answer: Yes\n\n"
                "Do you think trucks are a fruit? Answer: No\n\n"
                "Do you think robins are birds? Answer: Yes\n\n"
            )
            return few_shot_prefix + f"Do you think {item.noun1} are a {item.noun2}? Answer:"
        return ""
    
    def get_gen_prompt(item):
        """Get the generator prompt."""
        if task_config:
            result = task_config['make_prompt'](item, style='generator')
            return result.prompt if hasattr(result, 'prompt') else str(item)
        elif task in ['hypernym', 'hypernym-car']:
            return f"A {item.noun1} is a kind of"
        return ""
    
    # Pre-compute all prompts and metadata
    all_data = []
    for item in L_train_all:
        all_data.append({
            'noun1': get_noun1(item),
            'noun2': get_noun2(item),
            'completion': get_item_completion(item),
            'gen_prompt': get_gen_prompt(item),
            'val_prompt': get_val_prompt(item),
        })
    
    # Initialize results
    gen_scores = [None] * len(all_data)
    val_scores = [None] * len(all_data)
    
    # Convert yestoks/notoks to tensors for batched indexing
    yestoks_tensor = torch.tensor(yestoks, device=device)
    notoks_tensor = torch.tensor(notoks, device=device)
    
    with torch.no_grad():
        # === BATCHED VALIDATOR SCORING ===
        print("  Computing validator scores (batched)...")
        for batch_start in tqdm(range(0, len(all_data), batch_size), desc="Val scores"):
            batch_end = min(batch_start + batch_size, len(all_data))
            batch_prompts = [all_data[i]['val_prompt'] for i in range(batch_start, batch_end)]
            
            # Tokenize batch with left padding
            tokenizer.padding_side = 'left'
            encoded = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab]
            
            # Get probabilities at last position for each item
            # With left padding, last position is always the prediction position
            last_logits = logits[:, -1, :]  # [batch, vocab]
            probs = torch.softmax(last_logits, dim=-1)  # [batch, vocab]
            
            # Compute yes/no probabilities
            p_yes = probs[:, yestoks_tensor].sum(dim=-1)  # [batch]
            p_no = probs[:, notoks_tensor].sum(dim=-1)  # [batch]
            
            # Compute log-odds or log-prob
            if validator_log_odds:
                batch_val_scores = torch.log(p_yes + 1e-12) - torch.log(p_no + 1e-12)
            else:
                batch_val_scores = torch.log(p_yes + 1e-12)
            
            # Store results
            for i, score in enumerate(batch_val_scores.cpu().tolist()):
                val_scores[batch_start + i] = score
        
        # === BATCHED GENERATOR SCORING ===
        print("  Computing generator scores (batched)...")
        for batch_start in tqdm(range(0, len(all_data), batch_size), desc="Gen scores"):
            batch_end = min(batch_start + batch_size, len(all_data))
            batch_items = [all_data[i] for i in range(batch_start, batch_end)]
            
            if use_full_completion:
                # For full completion, we need prompt + completion together
                # Tokenize each separately to know completion boundaries
                batch_gen_scores = []
                for item in batch_items:
                    prompt = item['gen_prompt']
                    completion = " " + item['completion']
                    
                    # Tokenize prompt and full sequence
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                    full_text = prompt + completion
                    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
                    
                    # The completion tokens are those after the prompt
                    completion_start = len(prompt_ids)
                    completion_ids = full_ids[completion_start:]
                    
                    if len(completion_ids) == 0:
                        batch_gen_scores.append(float('-inf'))
                        continue
                    
                    # Forward pass on full sequence
                    input_tensor = torch.tensor([full_ids], device=device)
                    outputs = model(input_ids=input_tensor)
                    logits = outputs.logits[0]  # [seq_len, vocab]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Sum log probs for completion tokens
                    # logits[t] predicts token t+1, so for completion starting at position completion_start,
                    # we need log_probs[completion_start-1:completion_start-1+len(completion_ids)]
                    total_log_prob = 0.0
                    for i, tok_id in enumerate(completion_ids):
                        pos = completion_start - 1 + i
                        if pos < log_probs.shape[0]:
                            total_log_prob += log_probs[pos, tok_id].item()
                    
                    if length_normalize and len(completion_ids) > 0:
                        total_log_prob = total_log_prob / len(completion_ids)
                    
                    batch_gen_scores.append(total_log_prob)
                
                for i, score in enumerate(batch_gen_scores):
                    gen_scores[batch_start + i] = score
            else:
                # First token only - can be batched more efficiently
                batch_prompts = [item['gen_prompt'] for item in batch_items]
                batch_completions = [" " + item['completion'] for item in batch_items]
                
                # Get first token of each completion
                first_tokens = []
                for comp in batch_completions:
                    toks = tokenizer.encode(comp, add_special_tokens=False)
                    first_tokens.append(toks[0] if toks else 0)
                
                # Tokenize prompts
                tokenizer.padding_side = 'left'
                encoded = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_logits = outputs.logits[:, -1, :]
                log_probs = torch.log_softmax(last_logits, dim=-1)
                
                # Get log prob for each completion's first token
                for i, tok_id in enumerate(first_tokens):
                    gen_scores[batch_start + i] = log_probs[i, tok_id].item()
    
    # Build results
    results = []
    for i, data in enumerate(all_data):
        results.append({
            'noun1': data['noun1'],
            'noun2': data['noun2'],
            'gen_score': gen_scores[i],
            'val_score': val_scores[i],
        })
    
    model.train()
    return results


def save_tracked_scores(results, output_path):
    """Save tracked scores to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['noun1', 'noun2', 'gen_score', 'val_score'])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved tracked scores to {output_path}")


def get_tracking_base_filename(model_name, task, delta, train_g_or_d, use_all, split_type, alpha,
                                typicality_correction, length_normalize, use_full_completion,
                                nll_validator_weight, nll_generator_weight, force_same_x=False,
                                boost_initial_val=False):
    """Generate base filename for tracking logs (same as model save name but without epoch)."""
    direction_str = {'d': 'g2d', 'g': 'd2g', 'iter': 'iter', 'both': 'both'}[train_g_or_d]
    all_str = "-all" if use_all else ""
    alpha_str = f"-alpha{alpha}" if isinstance(alpha, (int, float)) else f"-alpha-{alpha}"
    typcorr_str = "-tc-online" if typicality_correction else ""  # tc = typicality correction, online = applied during training
    lenorm_str = "-lenorm" if length_normalize else ""
    full_completion_str = "-full-completion" if use_full_completion else ""
    nll_v_str = f"-nllv{nll_validator_weight}" if nll_validator_weight > 0 else ""
    nll_g_str = f"-nllg{nll_generator_weight}" if nll_generator_weight > 0 else ""
    force_same_x_str = "-force-same-x" if force_same_x else ""
    valboost_str = "-valboost" if boost_initial_val else ""
    
    base_name = (f"v5-{model_name.replace('/', '--')}-delta{delta}--{task}{all_str}"
                 f"--{direction_str}--{split_type}{alpha_str}{typcorr_str}{lenorm_str}"
                 f"{full_completion_str}{nll_v_str}{nll_g_str}{force_same_x_str}{valboost_str}")
    return base_name


def main(args):
    model_name = args.model
    task = args.task
    with_ref = args.with_ref
    num_epochs = args.num_epochs
    lr = args.learning_rate
    delta = args.delta
    #TODO set delta automatically based on data?
    total_samples = args.total_samples
    save_steps = args.save_steps
    use_all = args.all  # New flag for using all examples
    train_g_or_d = args.train_g_or_d
    split_type = args.split_type
    alpha = args.alpha  # New alpha parameter
    use_lora = args.lora
    gradient_checkpointing = args.gradient_checkpointing
    use_full_completion = not args.no_full_completion
    debug = args.debug
    nll_validator_weight = args.nll_validator_weight
    nll_generator_weight = args.nll_generator_weight
    use_wandb = not args.no_wandb
    validator_log_odds = args.validator_log_odds
    track_scores = args.track_scores
    track_scores_freq = args.track_scores_freq
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup tracking directory and base filename
    if track_scores:
        tracking_base_name = get_tracking_base_filename(
            model_name, task, delta, train_g_or_d, use_all, split_type, alpha,
            args.typicality_correction, args.length_normalize, use_full_completion,
            nll_validator_weight, nll_generator_weight, args.force_same_x,
            args.boost_initial_val
        )
        tracking_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "outputs", "training-logs")
        os.makedirs(tracking_dir, exist_ok=True)
        print(f"Score tracking enabled. Logs will be saved to: {tracking_dir}/{tracking_base_name}-step*.csv")

    WITH_REF = with_ref
    
    # Yes/No token variants for log-odds computation
    yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
    no_words = ["No", " No", "NO", "no", " no"]
    
    # Initialize wandb if enabled
    if use_wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            # Auto-generate run name from key parameters
            run_name = f"{task}-{train_g_or_d}-delta{delta}-nllv{nll_validator_weight}-nllg{nll_generator_weight}-lr{lr}"
        
        wandb.init(
            project="rankalign",
            name=run_name,
            config={
                "model": model_name,
                "task": task,
                "train_g_or_d": train_g_or_d,
                "delta": delta,
                "nll_validator_weight": nll_validator_weight,
                "nll_generator_weight": nll_generator_weight,
                "learning_rate": lr,
                "num_epochs": num_epochs,
                "total_samples": total_samples,
                "with_ref": with_ref,
                "use_all": use_all,
                "split_type": split_type,
                "alpha": alpha,
                "use_lora": use_lora,
                "gradient_checkpointing": gradient_checkpointing,
                "use_full_completion": use_full_completion,
                "single_token_data_only": args.single_token_data_only,
            }
        )
        print(f"Weights & Biases initialized: rankalign/{run_name}")
    
    # Compatibility check: --use-full-completion is not yet supported with --with_ref
    if use_full_completion and WITH_REF:
        raise ValueError("--use-full-completion is not yet compatible with --with_ref. "
                        "The reference model scoring needs to be updated for multi-token completions.")

    # Compatibility check: NLL weights not yet supported with 'both' mode
    #TODO need to add this later!
    if train_g_or_d == 'both' and (nll_validator_weight > 0 or nll_generator_weight > 0):
        raise ValueError("NLL weights (--nll_validator_weight, --nll_generator_weight) are not yet "
                        "supported with --train_g_or_d both. Use 'd' or 'g' mode instead.")

    if 'Instruct' in model_name or 'instruct' in model_name or '-it' in model_name:
        with_chat = True
        print(f"Detected instruct model: {model_name}")
        print("Using chat template formatting for prompts")
        disc_shots = "zero"
        space_prefix = ""
    else:
        with_chat = False
        disc_shots = "few"
        space_prefix = " "
        print(f"Using standard formatting for model: {model_name}")

    has_system_role = False
    if 'llama' in model_name.lower():
        has_system_role = True
        print("Model has system role!")

    # Define device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def load_model_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use memory-efficient loading for large models
        if 'gemma' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                attn_implementation="eager", 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif 'llama' in model_name.lower() or '8B' in model_name or '7B' in model_name:
            # For large Llama models, use more aggressive memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
                # Note: flash_attention_2 requires separate installation
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

    tokenizer, model = load_model_tokenizer(model_name)
    # Note: device_map="auto" in load_model_tokenizer handles device placement
    
    # Compute yes/no token IDs for log-odds computation
    yestoks = [tokenizer.encode(w)[-1] for w in yes_words]
    notoks = [tokenizer.encode(w)[-1] for w in no_words]
    if validator_log_odds:
        print(f"Using log-odds for validator: yestoks={yestoks}, notoks={notoks}")
    
    # Conditionally add LoRA for memory-efficient fine-tuning
    if use_lora:
        print("Setting up LoRA for memory-efficient fine-tuning...")
        lora_config = LoraConfig(
            r=16,  # Low-rank dimension
            lora_alpha=32,  # LoRA scaling parameter
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Llama target modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Show how many parameters we're actually training
    else:
        print("Using full model fine-tuning (no LoRA)")
        # For full model fine-tuning, ensure model is on correct device if device_map didn't handle it
        if not hasattr(model, 'hf_device_map'):
            model.to(device)
    
    # Enable gradient checkpointing to save memory (optional)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Gradient checkpointing disabled")

    if WITH_REF:
        print("Loading reference model with memory optimizations...")
        if 'gemma' in model_name.lower():
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name, 
                attn_implementation="eager", 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif 'llama' in model_name.lower() or '8B' in model_name or '7B' in model_name:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
    else:
        model_ref = None

    # see if padding right works??
    tokenizer.padding_side = 'left'

    # assume label of 1 meaning left < right in ground truth

    # NOTE first use ground truth ranking from generator.
    # Will now use ranking loss on *discriminator* prompts to try to match it!

    # Check task registry first (for new extensible tasks)
    task_config = get_task(task)
    use_v2 = not args.no_v2
    if task_config is not None:
        # NEW PATH: Use registered task configuration
        L_train, L_test = task_config['load_data'](seed=0, split_type=split_type, v2=use_v2)
    # LEGACY PATH: Existing task implementations (unchanged)
    elif task=='hypernym':
        L = utils.load_noun_pair_data()
        if split_type=='hyper':
            L_train, L_test = utils.split_train_test_no_overlap(L, seed=0)
        elif split_type=='random':
            L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
        elif split_type=='both':
            L_train, L_test = utils.split_train_test_no_overlap_both(L, seed=2)
        else:
            raise ValueError("Wrong value for split-type")
        #L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
        #L_train, L_test = utils.split_train_test_no_overlap(L, seed=0)
        #L_train, L_test = utils.split_train_test_no_overlap_both(L)
    elif task=='hypernym-car':
        # Pre-split balanced dataset for "cars are a kind of X"
        L_train, L_test = utils.load_hypernym_car_data()
    elif task=='trivia-qa':
        #USE SUBSET FOR NOW
        #L_train =  L['train'].shuffle(seed=42).select(range(3000))
        #L_test = L['validation'].shuffle(seed=42).select(range(1000))
        L_train, L_test, _ = utils.get_L_prompt('trivia-qa', split_type, seed=0)
    elif task=='swords':
        L_train, L_test = utils.load_swords_data(seed=0)
    elif task=='lambada':
        #L_train, L_test = utils.load_lambada_data(seed=0)
        # experiment with negatives -- recent version of get_L_prompt does this
        L_train, L_test, _ = utils.get_L_prompt('lambada', split_type, seed=0)
    elif task=='ifeval':
        L_train, L_test = utils.load_ifeval_data(seed=0)
    elif task=='collie':
        L_train, L_test = utils.load_collie_data(seed=0)
    else:
        raise NotImplementedError("Task not implemented!")

    # Filter for single-token completions if requested
    if args.single_token_data_only:
        print(f"Original L_train size: {len(L_train)}")
        # Determine the appropriate make_prompt function for the task
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            make_prompt_fn = task_config['make_prompt']
        # LEGACY PATH: Existing task implementations (unchanged)
        elif task in ['hypernym', 'hypernym-car']:
            make_prompt_fn = make_prompt_hypernymy
        elif task == 'trivia-qa':
            make_prompt_fn = make_prompt_triviaqa
        elif task == 'swords':
            make_prompt_fn = make_prompt_swords
        elif task == 'lambada':
            make_prompt_fn = make_prompt_lambada
        else:
            raise ValueError(f"Task {task} not supported for single_token_data_only filtering")
        
        filtered_L_train = []
        for item in L_train:
            gen_prompt_obj = make_prompt_fn(item, style="generator", shots='zero')
            completion_tokens = tokenizer.encode(gen_prompt_obj.completion, add_special_tokens=False)
            if len(completion_tokens) == 1:
                filtered_L_train.append(item)
        L_train = filtered_L_train
        print(f"Filtered to single-token completions: {len(L_train)}")

    print("Computing log-probabilities on the fly...")
    print(f"Using device: {device}")

    if train_g_or_d=='d':
        # Assume the generator is absolutely correct and try to match it.
        gold_prompt_style = 'generator'
        gold_prompt_shots = 'zero' #always using zero-shot for generator prompts

        tune_prompt_style = 'discriminator'
        tune_prompt_shots = disc_shots #depends on whether is instruct tuned model

    elif train_g_or_d =='g':
        # Assume the discriminator is absolutely correct and try to match generator to it.
        gold_prompt_style = 'discriminator'
        gold_prompt_shots = disc_shots #always using zero-shot for generator prompts

        tune_prompt_style = 'generator'
        tune_prompt_shots = 'zero'

    elif train_g_or_d == 'both':
        # For this mode, we train on a combination of generator and discriminator prompts
        # NOTE: gold_prompt_style and tune_prompt_style might be misleading in this setting.
        gold_prompt_style = 'generator'
        gold_prompt_shots = 'zero'
        
        tune_prompt_style = 'discriminator'
        tune_prompt_shots = disc_shots

    elif train_g_or_d == 'i':
        raise NotImplementedError("TODO implement this")
    else:
        raise NotImplementedError("train_g_or_d needs to be 'g', 'd', 'both', or 'i'.")


    # Check task registry first (for new extensible tasks)
    task_config = get_task(task)
    if task_config is not None:
        # NEW PATH: Use registered task configuration
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if task_config['get_label'](i) == 'yes']
        
        # Generate gold prompts using task's make_prompt function
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(
            task_config['make_prompt'], L_train_all, tokenizer,
            style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None
        )
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for gold prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            if train_g_or_d == 'd':
                target_text = space_prefix + task_config['get_completion'](L_train_all[idx]).strip()
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d == 'g':
                target_text = space_prefix + "Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d == 'both':
                target_text_d = space_prefix + task_config['get_completion'](L_train_all[idx]).strip()
                target_tokens_d = tokenizer.encode(target_text_d)
                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")

            if use_full_completion:
                if train_g_or_d == 'both':
                    log_prob_d = get_completion_token_logprobs(prompt, target_text_d, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    log_prob_g = get_completion_token_logprobs(prompt, target_text_g, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    total_log_prob_d = float(log_prob_d.sum().item())
                    total_log_prob_g = float(log_prob_g.sum().item())
                    logprobs_last_layer.append((total_log_prob_d, total_log_prob_g))
                else:
                    log_prob = get_completion_token_logprobs(prompt, target_text, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    total_log_prob = float(log_prob.sum().item())
                    logprobs_last_layer.append(total_log_prob)
            else:
                probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                if train_g_or_d == 'both':
                    ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                    ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                    # Assert that heuristic matches correct approach
                    ind_d_correct = tokenizer.encode(target_text_d, add_special_tokens=False)[0]
                    ind_g_correct = tokenizer.encode(target_text_g, add_special_tokens=False)[0]
                    assert ind_d == ind_d_correct, f"Token index mismatch (d): heuristic={ind_d}, correct={ind_d_correct}, target_text='{target_text_d}'"
                    assert ind_g == ind_g_correct, f"Token index mismatch (g): heuristic={ind_g}, correct={ind_g_correct}, target_text='{target_text_g}'"
                    log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                    log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                    logprobs_last_layer.append((log_prob_d, log_prob_g))
                else:
                    ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                    # Assert that heuristic matches correct approach (tokenize without special tokens)
                    ind_correct = tokenizer.encode(target_text, add_special_tokens=False)[0]
                    assert ind == ind_correct, f"Token index mismatch: heuristic={ind}, correct={ind_correct}, target_text='{target_text}', tokens_with_special={target_tokens}, tokens_without_special={tokenizer.encode(target_text, add_special_tokens=False)}"
                    log_prob = math.log(probs[ind].item() + 1e-12)
                    logprobs_last_layer.append(log_prob)

        # Generate tune prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(
            task_config['make_prompt'], L_train_all, tokenizer,
            style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None
        )

    # LEGACY PATH: Existing task implementations (unchanged)
    elif task in ['hypernym', 'hypernym-car']:
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.taxonomic == "yes"]
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_hypernymy, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for presumed "gold truth" prompts (when training discriminator, these are generator prompts)
        # if trainin disc, log_probs_last_layer_pos are for generator prompt
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            # Get the log probability for the target token (noun2)
            # For hypernymy, we want the probability of the noun2 token
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx].noun2
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx].noun2
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            
            if use_full_completion:
                if train_g_or_d == 'both':
                    log_prob_d = get_completion_token_logprobs(prompt, target_text_d, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    log_prob_g = get_completion_token_logprobs(prompt, target_text_g, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    total_log_prob_d = float(log_prob_d.sum().item())
                    total_log_prob_g = float(log_prob_g.sum().item())
                    logprobs_last_layer.append((total_log_prob_d, total_log_prob_g))
                else:
                    log_prob = get_completion_token_logprobs(prompt, target_text, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    total_log_prob = float(log_prob.sum().item())
                    logprobs_last_layer.append(total_log_prob)
            else:
                probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                if train_g_or_d == 'both':
                    ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                    ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                    # Assert that heuristic matches correct approach
                    ind_d_correct = tokenizer.encode(target_text_d, add_special_tokens=False)[0]
                    ind_g_correct = tokenizer.encode(target_text_g, add_special_tokens=False)[0]
                    assert ind_d == ind_d_correct, f"Token index mismatch (d): heuristic={ind_d}, correct={ind_d_correct}, target_text='{target_text_d}'"
                    assert ind_g == ind_g_correct, f"Token index mismatch (g): heuristic={ind_g}, correct={ind_g_correct}, target_text='{target_text_g}'"
                    log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                    log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                    logprobs_last_layer.append((log_prob_d, log_prob_g))
                    #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
                else:
                    ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                    # Assert that heuristic matches correct approach
                    ind_correct = tokenizer.encode(target_text, add_special_tokens=False)[0]
                    assert ind == ind_correct, f"Token index mismatch: heuristic={ind}, correct={ind_correct}, target_text='{target_text}', tokens_with_special={target_tokens}, tokens_without_special={tokenizer.encode(target_text, add_special_tokens=False)}"
                    log_prob = math.log(probs[ind].item() + 1e-12)
                    logprobs_last_layer.append(log_prob)

        # Generate discriminator prompts if train_g_or_d == 'd'.  Previously was p_train_disc
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_hypernymy, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]

    elif task=='trivia-qa':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i['correct']=='yes']

        #L_train_all = L_train  # Already using all examples for trivia-qa
        # Generate generator prompts
        #p_train_gen, hf_train_gen, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='generator', shots='zero', both=None)
        #prompts_gen = [i.prompt for i in p_train_gen]
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for generator prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
            if train_g_or_d=='d':
                # Get the log probability for the target token (answer)
                target_text = space_prefix + L_train_all[idx]['answers'][0].capitalize()
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx]['answers'][0].capitalize()
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                # Assert that heuristic matches correct approach
                ind_d_correct = tokenizer.encode(target_text_d, add_special_tokens=False)[0]
                ind_g_correct = tokenizer.encode(target_text_g, add_special_tokens=False)[0]
                assert ind_d == ind_d_correct, f"Token index mismatch (d): heuristic={ind_d}, correct={ind_d_correct}, target_text='{target_text_d}'"
                assert ind_g == ind_g_correct, f"Token index mismatch (g): heuristic={ind_g}, correct={ind_g_correct}, target_text='{target_text_g}'"
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                # Assert that heuristic matches correct approach
                ind_correct = tokenizer.encode(target_text, add_special_tokens=False)[0]
                assert ind == ind_correct, f"Token index mismatch: heuristic={ind}, correct={ind_correct}, target_text='{target_text}', tokens_with_special={target_tokens}, tokens_without_special={tokenizer.encode(target_text, add_special_tokens=False)}"
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)

        # Generate discriminator prompts
        #p_train_disc, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)

    elif task=='swords':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.synonym=='yes']
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for generator prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
            # Get the log probability for the target token (replacement)
            #target_text = space_prefix + L_train_all[idx].replacement
            #target_tokens = tokenizer.encode(target_text)
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx].replacement
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx].replacement
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                # Assert that heuristic matches correct approach
                ind_d_correct = tokenizer.encode(target_text_d, add_special_tokens=False)[0]
                ind_g_correct = tokenizer.encode(target_text_g, add_special_tokens=False)[0]
                assert ind_d == ind_d_correct, f"Token index mismatch (d): heuristic={ind_d}, correct={ind_d_correct}, target_text='{target_text_d}'"
                assert ind_g == ind_g_correct, f"Token index mismatch (g): heuristic={ind_g}, correct={ind_g_correct}, target_text='{target_text_g}'"
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                # Assert that heuristic matches correct approach
                ind_correct = tokenizer.encode(target_text, add_special_tokens=False)[0]
                assert ind == ind_correct, f"Token index mismatch: heuristic={ind}, correct={ind_correct}, target_text='{target_text}', tokens_with_special={target_tokens}, tokens_without_special={tokenizer.encode(target_text, add_special_tokens=False)}"
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)
        # Generate discriminator prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]

    elif task=='lambada':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i['correct']=='yes']


        #L_train_all = L_train  # Already using all examples for lambada
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
 
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for generator prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
            # Get the log probability for the target token (final_word)
            #target_text = space_prefix + L_train_all[idx]['final_word']
            #target_tokens = tokenizer.encode(target_text)
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx]['final_word']
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx]['final_word']
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                # Assert that heuristic matches correct approach
                ind_d_correct = tokenizer.encode(target_text_d, add_special_tokens=False)[0]
                ind_g_correct = tokenizer.encode(target_text_g, add_special_tokens=False)[0]
                assert ind_d == ind_d_correct, f"Token index mismatch (d): heuristic={ind_d}, correct={ind_d_correct}, target_text='{target_text_d}'"
                assert ind_g == ind_g_correct, f"Token index mismatch (g): heuristic={ind_g}, correct={ind_g_correct}, target_text='{target_text_g}'"
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                # Assert that heuristic matches correct approach
                ind_correct = tokenizer.encode(target_text, add_special_tokens=False)[0]
                assert ind == ind_correct, f"Token index mismatch: heuristic={ind}, correct={ind_correct}, target_text='{target_text}', tokens_with_special={target_tokens}, tokens_without_special={tokenizer.encode(target_text, add_special_tokens=False)}"
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)
        # Generate discriminator prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
    elif task=='ifeval':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.correct]
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_ifeval, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for presumed "gold truth" prompts (when training discriminator, these are generator prompts)
        # if trainin disc, log_probs_last_layer_pos are for generator prompt
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            # Get the log probability for the target 
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx]['response']
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx]['response']
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")

            if not use_full_completion:
                raise ValueError("must use full completion for ifeval task")
            
            if train_g_or_d == 'both':
                log_prob_d = get_completion_token_logprobs(prompt, target_text_d, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                log_prob_g = get_completion_token_logprobs(prompt, target_text_g, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                total_log_prob_d = float(log_prob_d.sum().item())
                total_log_prob_g = float(log_prob_g.sum().item())
                logprobs_last_layer.append((total_log_prob_d, total_log_prob_g))
            else:
                log_prob = get_completion_token_logprobs(prompt, target_text, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                total_log_prob = float(log_prob.sum().item())
                logprobs_last_layer.append(total_log_prob)

        # Generate discriminator prompts if train_g_or_d == 'd'.  Previously was p_train_disc
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_ifeval, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
    elif task=='collie':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.correct]
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_collie, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for presumed "gold truth" prompts (when training discriminator, these are generator prompts)
        # if trainin disc, log_probs_last_layer_pos are for generator prompt
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            # Get the log probability for the target 
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx]['generated']
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx]['generated']
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")

            if not use_full_completion:
                raise ValueError("must use full completion for collie task")
            
            if train_g_or_d == 'both':
                log_prob_d = get_completion_token_logprobs(prompt, target_text_d, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                log_prob_g = get_completion_token_logprobs(prompt, target_text_g, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                total_log_prob_d = float(log_prob_d.sum().item())
                total_log_prob_g = float(log_prob_g.sum().item())
                logprobs_last_layer.append((total_log_prob_d, total_log_prob_g))
            else:
                log_prob = get_completion_token_logprobs(prompt, target_text, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                total_log_prob = float(log_prob.sum().item())
                logprobs_last_layer.append(total_log_prob)

        # Generate discriminator prompts if train_g_or_d == 'd'.  Previously was p_train_disc
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_collie, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
    else:
        raise ValueError("Task unsupported!")

    # Compute typicality scores if requested (for ALL modes)
    # Note: typicality_scores will be used during training to correct generator scores
    # For pair selection, we only apply the correction to logprobs_last_layer in 'd' and 'both' modes
    typicality_scores = None  # Initialize to None (will be list if computed)
    
    if args.typicality_correction:
        print("\n" + "="*60)
        print("COMPUTING TYPICALITY SCORES")
        print("="*60)
        
        # Extract completions based on task
        completions = []
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            completions = [task_config['get_completion'](item).strip() for item in L_train_all]
        # LEGACY PATH: Existing task implementations (unchanged)
        elif task in ['hypernym', 'hypernym-car']:
            completions = [item.noun2 for item in L_train_all]
        elif task == 'trivia-qa':
            completions = [item['answers'][0] for item in L_train_all]
        elif task == 'swords':
            completions = [item.replacement for item in L_train_all]
        elif task == 'lambada':
            completions = [item['final_word'] for item in L_train_all]
        elif task == 'ifeval':
            completions = [item['response'] for item in L_train_all]
        elif task == 'collie':
            completions = [item['generated'] for item in L_train_all]
        else:
            raise ValueError(f"Task {task} not supported for typicality correction")
        
        # Load GPT-2 for typicality computation
        print("\nLoading GPT-2 for typicality correction...")
        tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
        model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
        model_gpt2 = model_gpt2.to(device)
        model_gpt2.eval()
        print(f"  ✓ GPT-2 loaded on {device}")
        
        # Compute GPT-2 typicality scores
        typicality_scores = compute_gpt2_typicality(completions, tokenizer_gpt2, model_gpt2, device)
        print(f"  Computed typicality scores for {len(typicality_scores)} examples")
        print(f"  Typicality mean: {sum(typicality_scores)/len(typicality_scores):.4f}")
        
        # Apply correction to logprobs_last_layer ONLY for pair selection in 'd' and 'both' modes
        # (In 'g' mode, logprobs_last_layer contains validator scores, not generator scores)
        if train_g_or_d in ['d', 'both']:
            print("\nApplying correction to pair selection scores (for 'd'/'both' modes)")
            if train_g_or_d == 'both':
                # For 'both' mode, logprobs_last_layer contains tuples (log_prob_d, log_prob_g)
                # We correct log_prob_d (generator logprob for discriminator path)
                logprobs_original = logprobs_last_layer.copy()
                logprobs_last_layer = [(lp[0] - typicality_scores[i], lp[1]) for i, lp in enumerate(logprobs_last_layer)]
                original_means_d = sum([lp[0] for lp in logprobs_original]) / len(logprobs_original)
                corrected_means_d = sum([lp[0] for lp in logprobs_last_layer]) / len(logprobs_last_layer)
                print(f"  Original generator mean (d): {original_means_d:.4f}")
                print(f"  Corrected generator mean (d): {corrected_means_d:.4f}")
            else:
                # For 'd' mode, logprobs_last_layer is just a list of floats (generator scores)
                logprobs_original = logprobs_last_layer.copy()
                logprobs_last_layer = [lp - typicality_scores[i] for i, lp in enumerate(logprobs_last_layer)]
                original_mean = sum(logprobs_original) / len(logprobs_original)
                corrected_mean = sum(logprobs_last_layer) / len(logprobs_last_layer)
                print(f"  Original generator mean: {original_mean:.4f}")
                print(f"  Corrected generator mean: {corrected_mean:.4f}")
            print(f"  Correction applied to {len(logprobs_last_layer)} examples for pair selection")
        else:
            print("\n  (In 'g' mode: typicality will be applied during training, not pair selection)")
        
        # Clean up GPT-2 model
        del model_gpt2, tokenizer_gpt2
        torch.cuda.empty_cache()
        
        print("="*60 + "\n")

    # Compute val_boost_theta if requested
    # This shifts validator scores so the optimal classification threshold is 0
    val_boost_theta = 0.0  # Default: no boost
    
    if args.boost_initial_val:
        print("\n" + "="*60)
        print("COMPUTING VALIDATOR BOOST (--boost-initial-val)")
        print("="*60)
        
        # Extract validator scores based on mode
        if train_g_or_d == 'g':
            # In 'g' mode, logprobs_last_layer contains validator scores directly
            val_scores = logprobs_last_layer
        elif train_g_or_d == 'both':
            # In 'both' mode, logprobs_last_layer contains tuples (gen_score, val_score)
            val_scores = [lp[1] for lp in logprobs_last_layer]
        elif train_g_or_d == 'd':
            # In 'd' mode, validator scores are not precomputed
            # We need to compute them here for the purpose of finding the optimal threshold
            print("  Computing validator scores for 'd' mode...")
            val_scores = []
            target_text_yes = space_prefix + "Yes"
            target_tokens_yes = tokenizer.encode(target_text_yes)
            target_token_yes = target_tokens_yes[0] if len(target_tokens_yes) == 1 else target_tokens_yes[1]
            
            model.eval()
            with torch.no_grad():
                for idx in tqdm(range(len(L_train_all)), desc="Computing validator scores"):
                    # Get the discriminator prompt (which asks Yes/No)
                    prompt = p_train_tune[idx].prompt
                    
                    # Compute log P("Yes") using get_final_logit_prob
                    # Note: get_final_logit_prob returns full log-prob distribution [vocab_size]
                    log_probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat, has_system_role=has_system_role)
                    log_prob_yes = log_probs[target_token_yes].item()
                    val_scores.append(log_prob_yes)
            # Note: model.train() called later in training loop (line ~1980)
        else:
            raise ValueError(f"Unknown mode: {train_g_or_d}")
        
        # Get ground truth labels
        labels = []
        for item in L_train_all:
            # Check task registry first (for new extensible tasks)
            task_config_boost = get_task(task)
            if task_config_boost is not None:
                label = task_config_boost['get_label'](item)
            elif task in ['hypernym', 'hypernym-car']:
                label = item.taxonomic.strip().lower()
            elif task == 'trivia-qa':
                label = item['correct'].strip().lower()
            elif task == 'swords':
                label = item.synonym.strip().lower()
            elif task == 'lambada':
                label = item['correct'].strip().lower()
            elif task == 'ifeval':
                label = item['correct'].strip().lower()
            elif task == 'collie':
                label = item['correct'].strip().lower() if 'correct' in item else 'yes'
            else:
                raise ValueError(f"Task {task} not supported for boost_initial_val")
            labels.append(1 if label == 'yes' else 0)
        
        # Compute optimal threshold
        optimal_threshold, best_accuracy = compute_optimal_threshold(val_scores, labels)
        val_boost_theta = -optimal_threshold
        
        print(f"  Validator scores: min={min(val_scores):.4f}, max={max(val_scores):.4f}, mean={sum(val_scores)/len(val_scores):.4f}")
        print(f"  Labels: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Best accuracy at threshold: {best_accuracy:.4f}")
        print(f"  val_boost_theta (= -threshold): {val_boost_theta:.4f}")
        print("="*60 + "\n")

    if with_chat and has_system_role:
        # Process discriminator prompts (p_train_tune)
        ms_tune = [ [ {"role": "system", "content": "Answer directly without explanation."},  {"role": "user", "content": i.prompt.strip()}, {"role": "model", "content": i.completion.strip()} ] for i in p_train_tune]
        toks_tune = tokenizer.apply_chat_template(ms_tune, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
        max_context_length = toks_tune.shape[1]
        
        # If mode is 'both', also process generator prompts (p_train_gold) and take the maximum
        if train_g_or_d == 'both':
            ms_gold = [ [ {"role": "system", "content": "Answer directly without explanation."},  {"role": "user", "content": i.prompt.strip()}, {"role": "model", "content": i.completion.strip()} ] for i in p_train_gold]
            toks_gold = tokenizer.apply_chat_template(ms_gold, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
            max_context_length = max(max_context_length, toks_gold.shape[1])
    elif with_chat:
        # Process discriminator prompts (p_train_tune)
        ms_tune = [ [ {"role": "user", "content": i.prompt.strip()}, {"role": "model", "content": i.completion.strip()} ] for i in p_train_tune]
        toks_tune = tokenizer.apply_chat_template(ms_tune, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
        max_context_length = toks_tune.shape[1]
        
        # If mode is 'both', also process generator prompts (p_train_gold) and take the maximum
        if train_g_or_d == 'both':
            ms_gold = [ [ {"role": "user", "content": i.prompt.strip()}, {"role": "model", "content": i.completion.strip()} ] for i in p_train_gold]
            toks_gold = tokenizer.apply_chat_template(ms_gold, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
            max_context_length = max(max_context_length, toks_gold.shape[1])
    else:
        #TODO later should make this cleaner in utils.make_and_format_data
        max_context_length = len(hf_train[0]['input_ids'])
        if train_g_or_d == 'both':
            max_context_length = max(len(hf_train_gold[0]['input_ids']), max_context_length)
    print("MAX CONTEXT LENGTH: ", max_context_length)

    # Prepare typicality scores for inclusion in Z (use zeros if not computed)
    typ_scores_for_z = typicality_scores if typicality_scores is not None else [0.0] * len(L_train_all)
    
    if train_g_or_d == 'both':
        # Create tuples of (discriminator_prompt, generator_prompt, logprobs, typicality)
        # Note: logprobs_last_layer contains tuples of (log_prob_d, log_prob_g)
        Z = list(zip(p_train_tune, p_train_gold, logprobs_last_layer, typ_scores_for_z))
        
        # Sort based on discriminator logprob (first element of the logprobs tuple)
        Z = sorted(Z, key=lambda i: i[2][0])  # Using i[2][0] to get the discriminator logprob

        # Calculate delta based on range of discriminator logprobs
        min_logprob = Z[0][2][0]  # Minimum discriminator logprob
        max_logprob = Z[-1][2][0]  # Maximum discriminator logprob

        print(f"Delta (minimum separation): {delta}")
        if delta!=0:
            NN = (max_logprob - min_logprob) / delta
            print(f"NN: {NN}")
        print(f"Min logprob: {min_logprob}")
        print(f"Max logprob: {max_logprob}")

        if args.force_same_x:
            # Group indices by generator prompt (p_train_gold.prompt)
            prompt_to_indices = defaultdict(list)
            for idx, z in enumerate(Z):
                gen_prompt = z[1].prompt  # z[1] is p_train_gold
                prompt_to_indices[gen_prompt].append(idx)
            
            print(f"\n{'='*60}")
            print(f"FORCE-SAME-X MODE (both)")
            print(f"{'='*60}")
            print(f"Found {len(prompt_to_indices)} unique generator prompts")
            
            # For each group: create pairs, filter by delta
            prompt_to_valid_pairs = {}
            total_valid_pairs = 0
            for prompt, indices in prompt_to_indices.items():
                group_pairs = []
                for i, j in itertools.combinations(indices, 2):
                    logprob_i = Z[i][2][0]  # discriminator logprob
                    logprob_j = Z[j][2][0]
                    if abs(logprob_i - logprob_j) > delta:
                        # Ensure i has lower logprob than j
                        group_pairs.append((i, j) if logprob_i < logprob_j else (j, i))
                prompt_to_valid_pairs[prompt] = group_pairs
                total_valid_pairs += len(group_pairs)
            
            print(f"Total valid pairs (after delta filter): {total_valid_pairs}")
            
            # Sample proportionally from each group
            num_groups = len(prompt_to_valid_pairs)
            samples_per_group = total_samples // num_groups
            remainder = total_samples % num_groups
            
            print(f"\nPairs in train set per category:")
            pair_inds = []
            for i, (prompt, pairs) in enumerate(prompt_to_valid_pairs.items()):
                # Distribute remainder across first few groups
                n_samples = samples_per_group + (1 if i < remainder else 0)
                print(f"{prompt[:80]}...\t{n_samples}")
                if len(pairs) < n_samples:
                    raise ValueError(
                        f"Not enough pairs for prompt '{prompt[:60]}...': "
                        f"need {n_samples}, have {len(pairs)}. "
                        f"Try reducing --total_samples or --delta."
                    )
                pair_inds.extend(random.sample(pairs, n_samples))
            
            random.shuffle(pair_inds)
            
            # Debug: show sample pairs
            print(f"\n--- Sample pairs (first 3) ---")
            for pi, (i, j) in enumerate(pair_inds[:3]):
                print(f"Pair {pi+1}:")
                print(f"  Prompt: '{Z[i][1].prompt[:80]}...'")
                print(f"  Completion A: '{Z[i][1].completion}' (logprob={Z[i][2][0]:.3f})")
                print(f"  Completion B: '{Z[j][1].completion}' (logprob={Z[j][2][0]:.3f})")
            print(f"{'='*60}\n")
        else:
            indices = range(len(Z))
            pair_inds = list(itertools.product(indices, repeat=2))
            pair_inds = [i for i in pair_inds if i[0] < i[1]]
            pair_inds = random.sample(pair_inds, total_samples)
        
        # Create pairs with all the information
        pairs_ = [(Z[i[0]], Z[i[1]]) for i in pair_inds]
    else:
        #Z = list(zip(prompts_pos, gen_logprobs_last_layer))
        # Include L_train_all to access ground truth labels (e.g., .taxonomic)
        # Also include typicality scores (index 3)
        Z = list(zip(p_train_tune, logprobs_last_layer, L_train_all, typ_scores_for_z))
        Z = sorted(Z, key = lambda i: i[1])  # Sort by logprob (index 1)

        # Calculate delta based on range of logprobs
        min_logprob = Z[0][1]
        max_logprob = Z[-1][1]

        print(f"Delta (minimum separation): {delta}")
        if delta!=0:
            NN = (max_logprob - min_logprob) / delta
            print(f"NN: {NN}")
        print(f"Min logprob: {min_logprob}")
        print(f"Max logprob: {max_logprob}")

        if args.force_same_x:
            # Group indices by prompt (p_train_tune.prompt)
            prompt_to_indices = defaultdict(list)
            for idx, z in enumerate(Z):
                prompt = z[0].prompt  # z[0] is p_train_tune
                prompt_to_indices[prompt].append(idx)
            
            print(f"\n{'='*60}")
            print(f"FORCE-SAME-X MODE (train_g_or_d={train_g_or_d})")
            print(f"{'='*60}")
            print(f"Found {len(prompt_to_indices)} unique prompts")
            
            # For each group: create pairs, filter by delta
            prompt_to_valid_pairs = {}
            total_valid_pairs = 0
            for prompt, indices in prompt_to_indices.items():
                group_pairs = []
                for i, j in itertools.combinations(indices, 2):
                    logprob_i = Z[i][1]  # logprob is at index 1
                    logprob_j = Z[j][1]
                    if abs(logprob_i - logprob_j) > delta:
                        # Ensure i has lower logprob than j
                        group_pairs.append((i, j) if logprob_i < logprob_j else (j, i))
                prompt_to_valid_pairs[prompt] = group_pairs
                total_valid_pairs += len(group_pairs)
            
            print(f"Total valid pairs (after delta filter): {total_valid_pairs}")
            
            # Sample proportionally from each group
            num_groups = len(prompt_to_valid_pairs)
            samples_per_group = total_samples // num_groups
            remainder = total_samples % num_groups
            
            print(f"\nPairs in train set per category:")
            pair_inds = []
            for i, (prompt, pairs) in enumerate(prompt_to_valid_pairs.items()):
                # Distribute remainder across first few groups
                n_samples = samples_per_group + (1 if i < remainder else 0)
                print(f"{prompt[:80]}...\t{n_samples}")
                if len(pairs) < n_samples:
                    raise ValueError(
                        f"Not enough pairs for prompt '{prompt[:60]}...': "
                        f"need {n_samples}, have {len(pairs)}. "
                        f"Try reducing --total_samples or --delta."
                    )
                pair_inds.extend(random.sample(pairs, n_samples))
            
            random.shuffle(pair_inds)
            
            # Debug: show sample pairs
            print(f"\n--- Sample pairs (first 3) ---")
            for pi, (i, j) in enumerate(pair_inds[:3]):
                print(f"Pair {pi+1}:")
                print(f"  Prompt: '{Z[i][0].prompt[:80]}...'")
                print(f"  Completion A: '{Z[i][0].completion}' (logprob={Z[i][1]:.3f})")
                print(f"  Completion B: '{Z[j][0].completion}' (logprob={Z[j][1]:.3f})")
            print(f"{'='*60}\n")
        else:
            indices = range(len(Z))
            pair_inds = list(itertools.product(indices, repeat=2))
            pair_inds = [i for i in pair_inds if i[0] < i[1]]
            pair_inds = random.sample(pair_inds, total_samples)
        
        pairs_ = [(Z[i[0]], Z[i[1]]) for i in pair_inds]


    def format_with_inst(prompt):
        if has_system_role:
            message = [
                {"role": "system", "content": "Answer directly without explanation."},
                {"role": "user", "content": prompt},]
        else:
            message = [
                {"role": "user", "content": prompt},]
        toks = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt')[0]
        decoded = tokenizer.decode(toks[1:])
        
        # Assert: decode/re-encode should produce the same tokens
        # If this fails, there's a tokenization asymmetry that could cause training inconsistencies
        reencoded = tokenizer.encode(decoded, add_special_tokens=False, return_tensors='pt')[0]
        original_without_bos = toks[1:]
        assert torch.equal(reencoded, original_without_bos), (
            f"Decode/re-encode mismatch! "
            f"Original tokens (no BOS): {original_without_bos.tolist()}, "
            f"Re-encoded tokens: {reencoded.tolist()}, "
            f"Decoded text: '{decoded[:100]}...'"
        )
        
        return decoded

    def get_correct_answer(data_item, task):
        """Get the ground truth answer (Yes/No) for a data item based on task type."""
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            label = task_config['get_label'](data_item)
        # LEGACY PATH: Existing task implementations (unchanged)
        elif task in ['hypernym', 'hypernym-car']:
            label = data_item.taxonomic.strip().lower()
        elif task == 'trivia-qa':
            label = data_item['correct'].strip().lower()
        elif task == 'swords':
            label = data_item.synonym.strip().lower()
        elif task == 'lambada':
            label = data_item['correct'].strip().lower()
        elif task == 'ifeval':
            label = data_item['correct'].strip().lower()
        else:
            raise ValueError(f"Task {task} not supported for ground truth lookup")
        
        if label == 'yes':
            return space_prefix + "Yes"
        else:
            return space_prefix + "No"

    def get_generator_completion(data_item, task):
        """Get the generator completion (actual task answer) for a data item."""
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            return space_prefix + task_config['get_completion'](data_item).strip()
        # LEGACY PATH: Existing task implementations (unchanged)
        if task in ['hypernym', 'hypernym-car']:
            return space_prefix + data_item.noun2
        elif task == 'trivia-qa':
            return space_prefix + data_item['answers'][0]
        elif task == 'swords':
            return space_prefix + data_item.replacement
        elif task == 'lambada':
            return space_prefix + data_item['final_word']
        elif task == 'ifeval':
            return space_prefix + data_item['response']
        elif task == 'collie':
            return space_prefix + data_item['generated']
        else:
            raise ValueError(f"Task {task} not supported for generator completion lookup")

    def get_indicator(data_item, task):
        """Get indicator (1 if positive example, 0 if negative)."""
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            return task_config['get_indicator'](data_item)
        # LEGACY PATH: Existing task implementations (unchanged)
        if task in ['hypernym', 'hypernym-car']:
            label = data_item.taxonomic.strip().lower()
        elif task == 'trivia-qa':
            label = data_item['correct'].strip().lower()
        elif task == 'swords':
            label = data_item.synonym.strip().lower()
        elif task == 'lambada':
            label = data_item['correct'].strip().lower()
        elif task == 'ifeval':
            label = data_item['correct'].strip().lower()
        else:
            raise ValueError(f"Task {task} not supported for indicator lookup")
        
        return 1.0 if label == 'yes' else 0.0


    if train_g_or_d=='d':
        #NOTE in this case the tokens we are targeting are the "Yes" tokens in both cases.
        completion_text = space_prefix +"Yes"

        if with_chat:
             pairs = [
                 (
                     (format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)),  # prompts
                     (completion_text, completion_text),  # completion for ranking (always "Yes")
                     (get_correct_answer(pair[0][2], task), get_correct_answer(pair[1][2], task)),  # validator correct answers
                     (get_generator_completion(pair[0][2], task), get_generator_completion(pair[1][2], task)),  # generator completions
                     (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task)),  # indicators (1=positive, 0=negative)
                     (pair[0][3], pair[1][3])  # typicality scores
                 )
                 for pair in pairs_ if pair[1][1] - pair[0][1] > delta
             ]
        else:
            pairs = [
                (
                    (pair[0][0].prompt, pair[1][0].prompt),  # prompts
                    (completion_text, completion_text),  # completion for ranking (always "Yes")
                    (get_correct_answer(pair[0][2], task), get_correct_answer(pair[1][2], task)),  # validator correct answers
                    (get_generator_completion(pair[0][2], task), get_generator_completion(pair[1][2], task)),  # generator completions
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task)),  # indicators (1=positive, 0=negative)
                    (pair[0][3], pair[1][3])  # typicality scores
                )
                for pair in pairs_ if pair[1][1] - pair[0][1] > delta
            ]
    elif train_g_or_d=='g':
        #NOTE in this case the ranking is derived from the log-probs of Yes under both prompts but we are targetting
        # the log-odds (hopefully log-prob is fine here) of the *generator completion*, so not the same in each item of the pair!
        if with_chat:
            pairs = [
                (
                    (format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)),  # prompts
                    (pair[0][0].completion, pair[1][0].completion),  # completion for ranking (generator completions)
                    (get_correct_answer(pair[0][2], task), get_correct_answer(pair[1][2], task)),  # validator correct answers
                    (get_generator_completion(pair[0][2], task), get_generator_completion(pair[1][2], task)),  # generator completions
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task)),  # indicators (1=positive, 0=negative)
                    (pair[0][3], pair[1][3])  # typicality scores
                )
                for pair in pairs_ if pair[1][1] - pair[0][1] > delta
            ]
        else:
            pairs = [
                (
                    (pair[0][0].prompt, pair[1][0].prompt),  # prompts
                    (pair[0][0].completion, pair[1][0].completion),  # completion for ranking (generator completions)
                    (get_correct_answer(pair[0][2], task), get_correct_answer(pair[1][2], task)),  # validator correct answers
                    (get_generator_completion(pair[0][2], task), get_generator_completion(pair[1][2], task)),  # generator completions
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task)),  # indicators (1=positive, 0=negative)
                    (pair[0][3], pair[1][3])  # typicality scores
                )
                for pair in pairs_ if pair[1][1] - pair[0][1] > delta
            ]
    elif train_g_or_d == 'both':
        # For both mode, we create pairs for both generator and discriminator training
        # First create discriminator pairs (targeting "Yes" tokens)
        completion_text = space_prefix +"Yes"
        if with_chat:
            # Create pairs with both discriminator and generator prompts, applying chat formatting
            # NOTE verify fixed
            # Z structure: (p_train_tune, p_train_gold, logprobs, typicality) - typicality at index 3
            pairs = [
                (
                    ((format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)), (completion_text, completion_text)),  # discriminator pair
                    ((format_with_inst(pair[0][1].prompt), format_with_inst(pair[1][1].prompt)), (pair[0][1].completion, pair[1][1].completion)),  # generator pair
                    (pair[0][0].completion.strip().lower()   , pair[1][0].completion.strip().lower()   ),  # labels
                    (pair[0][3], pair[1][3])  # typicality scores
                ) for pair in pairs_ if pair[1][2][0] - pair[0][2][0] > delta
            ]
        else:
            # Create pairs with both discriminator and generator prompts
            # Z structure: (p_train_tune, p_train_gold, logprobs, typicality) - typicality at index 3
            pairs = [
                (
                    ((pair[0][0].prompt, pair[1][0].prompt), (completion_text, completion_text)),  # discriminator pair
                    ((pair[0][1].prompt, pair[1][1].prompt), (pair[0][1].completion, pair[1][1].completion)),  # generator pair
                    (pair[0][0].completion.strip().lower()   , pair[1][0].completion.strip().lower()   ),  # labels
                    (pair[0][3], pair[1][3])  # typicality scores
                ) for pair in pairs_ if pair[1][2][0] - pair[0][2][0] > delta
            ]

    else:
        raise ValueError("TODO!")

    print(pairs[0])
    print("\n\n")
    print(pairs[1])
    print("\n\nNum Samples: ", len(pairs))

    class PairwiseDataset(Dataset):
        def __init__(self, pairs, tokenizer, max_length=128, device='cuda', use_full_completion=False):
            """
            pairs: list of ((prompt_i, prompt_j), (token_i, token_j))
            tokenizer: Hugging Face tokenizer
            device: device to place tensors on
            """
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = device
            self.use_full_completion = use_full_completion

            # Debug print first pair
            #print("\nDebugging PairwiseDataset initialization:")
            #print("First pair:", pairs[0])
            #print("Token types:", type(pairs[0][1][0]), type(pairs[0][1][1]))
            #print("Tokens:", pairs[0][1][0], pairs[0][1][1])

            # Try to encode the tokens
            #print("\nTrying to encode tokens:")
            #try:
            #    print("Encoding first token:", tokenizer.encode(pairs[0][1][0]))
            #    print("Encoding second token:", tokenizer.encode(pairs[0][1][1]))
            #except Exception as e:
            #    print("Error encoding tokens:", e)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            if train_g_or_d == 'both':
                # 4-element structure: (disc_pair, gen_pair, labels, typicality)
                ((prompt_i_disc, prompt_j_disc), (completion_i_disc, completion_j_disc)), ((prompt_i_gen, prompt_j_gen), (completion_i_gen, completion_j_gen)), (label_i, label_j), (typicality_i, typicality_j) = self.pairs[idx]
                if not self.use_full_completion:
                    completion_i_disc = self.tokenizer.decode(self.tokenizer.encode(completion_i_disc)[-1])
                    completion_j_disc = self.tokenizer.decode(self.tokenizer.encode(completion_j_disc)[-1])
                    completion_i_gen = self.tokenizer.decode(self.tokenizer.encode(completion_i_gen)[-1])
                    completion_j_gen = self.tokenizer.decode(self.tokenizer.encode(completion_j_gen)[-1])
            else:
                # Unified 6-element pair structure: (prompts, ranking_completions, validator_correct, gen_completions, indicators, typicality)
                (prompt_i, prompt_j), (completion_i, completion_j), (correct_i, correct_j), (gen_completion_i, gen_completion_j), (indicator_i, indicator_j), (typicality_i, typicality_j) = self.pairs[idx]
                
                if not self.use_full_completion:
                    completion_i = self.tokenizer.decode(self.tokenizer.encode(completion_i)[-1])
                    completion_j = self.tokenizer.decode(self.tokenizer.encode(completion_j)[-1])
                    correct_i = self.tokenizer.decode(self.tokenizer.encode(correct_i)[-1])
                    correct_j = self.tokenizer.decode(self.tokenizer.encode(correct_j)[-1])
                    gen_completion_i = self.tokenizer.decode(self.tokenizer.encode(gen_completion_i)[-1])
                    gen_completion_j = self.tokenizer.decode(self.tokenizer.encode(gen_completion_j)[-1])
            # Debug print
            #print(f"\nProcessing item {idx}:")
            #print("Token types:", type(token_i), type(token_j))
            #print("Tokens:", token_i, token_j)

            #TODO: truncate the completion if not using full completion
            if train_g_or_d == 'both':
                # Tokenize discriminator prompts
                input_i_disc = prompt_i_disc + completion_i_disc
                input_j_disc = prompt_j_disc + completion_j_disc

                enc_i_disc = self.tokenizer(
                    input_i_disc,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                enc_j_disc = self.tokenizer(
                    input_j_disc,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Tokenize generator prompts
                input_i_gen = prompt_i_gen + completion_i_gen
                input_j_gen = prompt_j_gen + completion_j_gen

                enc_i_gen = self.tokenizer(
                    input_i_gen,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                enc_j_gen = self.tokenizer(
                    input_j_gen,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                # Tokenize completions

                token_i_disc = self.tokenizer.encode(completion_i_disc, add_special_tokens=False, return_tensors='pt')
                token_j_disc = self.tokenizer.encode(completion_j_disc, add_special_tokens=False, return_tensors='pt')

                token_i_gen = self.tokenizer.encode(completion_i_gen, add_special_tokens=False, return_tensors='pt')
                token_j_gen = self.tokenizer.encode(completion_j_gen, add_special_tokens=False, return_tensors='pt')
            # Get labels from prompts
            #    label_i = "yes" if "yes" in prompt_i.lower() else "no"
            #    label_j = "yes" if "yes" in prompt_j.lower() else "no"



            else:
                # Tokenize prompt i
                input_i = prompt_i + completion_i
                enc_i = self.tokenizer(
                    input_i,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                
                # Tokenize completion i
                token_i = self.tokenizer.encode(completion_i, add_special_tokens=False, return_tensors='pt')

                # Tokenize prompt j
                input_j = prompt_j + completion_j
                enc_j = self.tokenizer(
                    input_j,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                # Tokenize completion j
                token_j = self.tokenizer.encode(completion_j, add_special_tokens=False, return_tensors='pt')

                # Tokenize correct completions (for validator NLL loss)
                token_correct_i = self.tokenizer.encode(correct_i, add_special_tokens=False, return_tensors='pt')
                token_correct_j = self.tokenizer.encode(correct_j, add_special_tokens=False, return_tensors='pt')

                # Tokenize generator completions (for generator NLL loss)
                token_gen_i = self.tokenizer.encode(gen_completion_i, add_special_tokens=False, return_tensors='pt')
                token_gen_j = self.tokenizer.encode(gen_completion_j, add_special_tokens=False, return_tensors='pt')

                # DEBUG: Check for tokenization mismatch (enabled with --debug flag)
                if debug:
                    print("\n" + "="*60)
                    print("DEBUG: TOKENIZATION MISMATCH CHECK")
                    print("="*60)
                    print(f"Completion text: '{completion_i}'")
                    separately_tokenized_completion = token_i.squeeze().tolist()
                    if not isinstance(separately_tokenized_completion, list):
                        separately_tokenized_completion = [separately_tokenized_completion]
                    print(f"Separately tokenized completion: {separately_tokenized_completion}")
                    
                    # Find actual tokens in full sequence
                    full_tokens = enc_i['input_ids'].squeeze().tolist()
                    # Remove padding tokens (usually 0 or pad_token_id)
                    pad_id = self.tokenizer.pad_token_id
                    actual_tokens = [t for t in full_tokens if t != pad_id]
                    
                    # Get last N tokens where N = len(completion tokens)
                    comp_len = token_i.squeeze().shape[0] if token_i.squeeze().dim() > 0 else 1
                    actual_completion_tokens = actual_tokens[-comp_len:]
                    
                    print(f"Actual tokens at end of full sequence: {actual_completion_tokens}")
                    do_they_match = separately_tokenized_completion == actual_completion_tokens
                    print(f"Do they match? {do_they_match}")
                    if not do_they_match:
                        print("*** MISMATCH DETECTED! ***")
                        breakpoint()
                    
                    # Decode both to see what text they represent
                    print(f"\nDecoded separately tokenized: '{self.tokenizer.decode(token_i.squeeze())}'")
                    print(f"Decoded from full sequence: '{self.tokenizer.decode(actual_completion_tokens)}'")
                    print("="*60)

            if train_g_or_d != 'both':
                # Squeeze to remove the batch dimension (shape: [seq_len])
                item = {
                    'input_ids_i': enc_i['input_ids'].squeeze(0),
                    'attention_mask_i': enc_i['attention_mask'].squeeze(0),
                    'token_id_i': token_i.squeeze(0),
                    'token_correct_i': token_correct_i.squeeze(0),  # validator correct answer
                    'token_gen_i': token_gen_i.squeeze(0),  # generator completion
                    'indicator_i': torch.tensor(indicator_i, dtype=torch.float),  # 1 if positive, 0 if negative
                    'input_ids_j': enc_j['input_ids'].squeeze(0),
                    'attention_mask_j': enc_j['attention_mask'].squeeze(0),
                    'token_id_j': token_j.squeeze(0),
                    'token_correct_j': token_correct_j.squeeze(0),  # validator correct answer
                    'token_gen_j': token_gen_j.squeeze(0),  # generator completion
                    'indicator_j': torch.tensor(indicator_j, dtype=torch.float),  # 1 if positive, 0 if negative
                    'label': torch.tensor(1.0, dtype=torch.float),
                    'typicality_i': torch.tensor(typicality_i, dtype=torch.float),  # GPT-2 P(completion) for item i
                    'typicality_j': torch.tensor(typicality_j, dtype=torch.float),  # GPT-2 P(completion) for item j
                }
            else:
                item = {
                    'input_ids_i_disc': enc_i_disc['input_ids'].squeeze(0),
                    'attention_mask_i_disc': enc_i_disc['attention_mask'].squeeze(0),
                    'token_id_i_disc': token_i_disc.squeeze(0),
                    'input_ids_j_disc': enc_j_disc['input_ids'].squeeze(0),
                    'attention_mask_j_disc': enc_j_disc['attention_mask'].squeeze(0),
                    'token_id_j_disc': token_j_disc.squeeze(0),
                    'input_ids_i_gen': enc_i_gen['input_ids'].squeeze(0),
                    'attention_mask_i_gen': enc_i_gen['attention_mask'].squeeze(0),
                    'token_id_i_gen': token_i_gen.squeeze(0),
                    'input_ids_j_gen': enc_j_gen['input_ids'].squeeze(0),
                    'attention_mask_j_gen': enc_j_gen['attention_mask'].squeeze(0),
                    'token_id_j_gen': token_j_gen.squeeze(0),
                    'label_i': torch.tensor(1.0 if label_i == "yes" else 0.0, dtype=torch.float),
                    'label_j': torch.tensor(1.0 if label_j == "yes" else 0.0, dtype=torch.float),
                    'typicality_i': torch.tensor(typicality_i, dtype=torch.float),  # GPT-2 P(completion) for item i
                    'typicality_j': torch.tensor(typicality_j, dtype=torch.float),  # GPT-2 P(completion) for item j
                }
            return item


    #18 fine for zero-shot
    if use_full_completion:
        batch_size = 1 #TODO: allow actual batches
    else:
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            batch_sizes = task_config.get('batch_size', {'with_ref': 1, 'without_ref': 2})
            batch_size = batch_sizes['with_ref'] if with_ref else batch_sizes['without_ref']
        # LEGACY PATH: Existing task implementations (unchanged)
        elif with_ref:
            if task=='swords':
                batch_size = 2
            elif task=='trivia-qa':
                batch_size = 2
            elif task=='lambada':
                batch_size = 2
            elif task in ['hypernym', 'hypernym-car']:
                batch_size = 1 #4
            elif task == 'ifeval':
                batch_size = 1
            elif task =='collie':
                batch_size = 2
            else:
                raise ValueError("define batch size for this case")
        else:
            if task=='swords':
                batch_size = 1#6
            elif task=='trivia-qa':
                batch_size = 2#6
            elif task=='lambada':
                batch_size = 2#6
            elif task in ['hypernym', 'hypernym-car']:
                batch_size = 2#6#1  # Reduced from 32 to 1 for large models
            elif task == 'ifeval':
                batch_size = 1
            elif task =='collie':
                batch_size = 2
            else:
                raise ValueError("define batch size for this case")

    # if max_context_length > 90:
    #     max_context_length = 90

    dataset = PairwiseDataset(pairs, tokenizer, max_length=max_context_length, device=device, use_full_completion=use_full_completion)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("\n\nDone making dataloader\n\n")
    optimizer = AdamW(model.parameters(), lr=lr)

    losses = []
    global_step = 0

    # Track scores at step 0 (before any training)
    if track_scores:
        print(f"\n  [Step 0] Tracking initial scores for all datapoints...")
        task_config_for_tracking = get_task(task)
        tracked_results = track_all_scores(
            model, tokenizer, L_train_all, task, device, yestoks, notoks,
            length_normalize=args.length_normalize,
            use_full_completion=use_full_completion,
            task_config=task_config_for_tracking,
            validator_log_odds=True,  # Always use log-odds for tracking (consistent with eval.py)
            is_chat=with_chat,
            has_system_role=has_system_role
        )
        tracking_path = os.path.join(tracking_dir, f"{tracking_base_name}-step0.csv")
        save_tracked_scores(tracked_results, tracking_path)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            def sum_completion_logprobs(log_probs, token_ids, length_normalize=False):
                """
                log_probs: [batch, seq_len, vocab] - over input_ids (prompt + completion), left padded
                token_ids: [batch, completion_len] - only the completion tokens
                length_normalize: if True, divide sum by number of tokens
                
                Note: log_probs[t] predicts token at position t+1, so for completion tokens
                at positions [-C:], we need log_probs at positions [-(C+1):-1]
                """
                completion_log_probs = []
                batch_size, seq_len, vocab_size = log_probs.shape

                for b in range(batch_size):
                    # pick the slice for this batch element
                    # log_probs[t] predicts token[t+1], so we need positions [-(C+1):-1] to predict tokens [-C:]
                    comp_len = token_ids[b].size(0)
                    comp_log_probs = log_probs[b, -(comp_len+1):-1, :]  # [completion_len, vocab]
                    # gather the logprobs for the actual completion tokens
                    gathered = comp_log_probs.gather(1, token_ids[b].unsqueeze(-1)).squeeze(-1)
                    score = gathered.sum()
                    if length_normalize and comp_len > 0:
                        score = score / comp_len
                    completion_log_probs.append(score)

                return torch.stack(completion_log_probs)

            # Move all inputs to device
            if train_g_or_d == 'both':
                # Get discriminator inputs
                input_ids_i_disc = batch["input_ids_i_disc"].to(device)
                attention_mask_i_disc = batch["attention_mask_i_disc"].to(device)
                token_id_i_disc = batch["token_id_i_disc"].to(device)
                input_ids_j_disc = batch["input_ids_j_disc"].to(device)
                attention_mask_j_disc = batch["attention_mask_j_disc"].to(device)
                token_id_j_disc = batch["token_id_j_disc"].to(device)

                # Get generator inputs
                input_ids_i_gen = batch["input_ids_i_gen"].to(device)
                attention_mask_i_gen = batch["attention_mask_i_gen"].to(device)
                token_id_i_gen = batch["token_id_i_gen"].to(device)
                input_ids_j_gen = batch["input_ids_j_gen"].to(device)
                attention_mask_j_gen = batch["attention_mask_j_gen"].to(device)
                token_id_j_gen = batch["token_id_j_gen"].to(device)

                label_i = batch["label_i"].to(device)
                label_j = batch["label_j"].to(device)

                # Forward pass for discriminator prompts
                outputs_i_disc = model(input_ids=input_ids_i_disc, attention_mask=attention_mask_i_disc)
                outputs_j_disc = model(input_ids=input_ids_j_disc, attention_mask=attention_mask_j_disc)
                
                # Forward pass for generator prompts
                outputs_i_gen = model(input_ids=input_ids_i_gen, attention_mask=attention_mask_i_gen)
                outputs_j_gen = model(input_ids=input_ids_j_gen, attention_mask=attention_mask_j_gen)

                # Compute log probabilities
                log_probs_i_disc = F.log_softmax(outputs_i_disc.logits, dim=-1)
                log_probs_j_disc = F.log_softmax(outputs_j_disc.logits, dim=-1)
                log_probs_i_gen = F.log_softmax(outputs_i_gen.logits, dim=-1)
                log_probs_j_gen = F.log_softmax(outputs_j_gen.logits, dim=-1)

                # Get discriminator scores
                if validator_log_odds:
                    # Log-odds: log(sum P(yes_tokens)) - log(sum P(no_tokens))
                    # Look at position before completion (last position predicts first completion token)
                    def compute_logodds(log_probs, token_ids):
                        """Compute log-odds for yes vs no at the position predicting the completion."""
                        batch_size = log_probs.shape[0]
                        logodds_list = []
                        for b in range(batch_size):
                            comp_len = token_ids[b].size(0)
                            # Position that predicts first completion token
                            pred_pos = -(comp_len + 1)
                            probs_at_pos = torch.exp(log_probs[b, pred_pos, :])  # [vocab]
                            p_yes = probs_at_pos[yestoks].sum()
                            p_no = probs_at_pos[notoks].sum()
                            logodds = torch.log(p_yes + 1e-12) - torch.log(p_no + 1e-12)
                            logodds_list.append(logodds)
                        return torch.stack(logodds_list)
                    
                    score_i_disc = compute_logodds(log_probs_i_disc, token_id_i_disc)
                    score_j_disc = compute_logodds(log_probs_j_disc, token_id_j_disc)
                else:
                    # Log-probs: log(P(completion))
                    score_i_disc = sum_completion_logprobs(log_probs_i_disc, token_id_i_disc)   
                    score_j_disc = sum_completion_logprobs(log_probs_j_disc, token_id_j_disc)
                
                # Apply validator boost (shift scores so optimal threshold is 0)
                if args.boost_initial_val:
                    score_i_disc = score_i_disc + val_boost_theta
                    score_j_disc = score_j_disc + val_boost_theta
                
                # Generator always uses log-probs (completion can be multi-token)
                # Apply length normalization if flag is set
                score_i_gen = sum_completion_logprobs(log_probs_i_gen, token_id_i_gen, length_normalize=args.length_normalize)
                score_j_gen = sum_completion_logprobs(log_probs_j_gen, token_id_j_gen, length_normalize=args.length_normalize)
                
                # Apply typicality correction to generator scores (online, during training)
                if args.typicality_correction:
                    typicality_i = batch["typicality_i"].to(device)
                    typicality_j = batch["typicality_j"].to(device)
                    score_i_gen = score_i_gen - typicality_i
                    score_j_gen = score_j_gen - typicality_j

                # Use frozen reference model if needed
                if WITH_REF:
                    raise ValueError("Do LATER")
                else:
                    diff_ref = 0

                # Compute both G->V and V->G losses
                g2v_diff = score_j_disc - score_i_disc - diff_ref
                v2g_diff = score_j_gen - score_i_gen - diff_ref

                # Get alpha for each sample in the batch
                alphas = []
                for b in range(batch["input_ids_i_disc"].size(0)):
                    alpha_val = get_alpha(alpha, 
                                    score_i_gen[b].item(), score_j_gen[b].item(),
                                    score_i_disc[b].item(), score_j_disc[b].item(),
                                    label_i[b].item(), label_j[b].item())
                    alphas.append(alpha_val)
                alphas = torch.tensor(alphas, device=device)

                # Compute weighted loss
                g2v_loss = -torch.log(torch.sigmoid(g2v_diff) + 1e-12)
                v2g_loss = -torch.log(torch.sigmoid(v2g_diff) + 1e-12)
                preference_loss = (alphas * g2v_loss + (1 - alphas) * v2g_loss).mean()
                
                # Note: NLL loss not yet implemented for 'both' mode
                # Use 'd' or 'g' mode with --nll_validator_weight or --nll_generator_weight
                loss = preference_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                global_step += 1
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/preference_loss": preference_loss.item(),
                        "train/g2v_loss": g2v_loss.mean().item(),
                        "train/v2g_loss": v2g_loss.mean().item(),
                        "train/score_j_disc": score_j_disc.mean().item(),
                        "train/score_i_disc": score_i_disc.mean().item(),
                        "train/score_j_gen": score_j_gen.mean().item(),
                        "train/score_i_gen": score_i_gen.mean().item(),
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    })
                
                # Track scores for all datapoints at specified frequency (both mode)
                if track_scores and global_step % track_scores_freq == 0:
                    print(f"\n  [Step {global_step}] Tracking scores for all datapoints...")
                    
                    task_config_for_tracking = get_task(task)
                    tracked_results = track_all_scores(
                        model, tokenizer, L_train_all, task, device, yestoks, notoks,
                        length_normalize=args.length_normalize,
                        use_full_completion=use_full_completion,
                        task_config=task_config_for_tracking,
                        validator_log_odds=True,  # Always use log-odds for tracking (consistent with eval.py)
                        is_chat=with_chat,
                        has_system_role=has_system_role
                    )
                    
                    tracking_path = os.path.join(tracking_dir, f"{tracking_base_name}-step{global_step}.csv")
                    save_tracked_scores(tracked_results, tracking_path)
                    
                    # Decode completions for 'both' mode
                    completion_i_disc_text = tokenizer.decode(token_id_i_disc[0] if token_id_i_disc.dim() > 1 else token_id_i_disc, skip_special_tokens=True).strip()
                    completion_j_disc_text = tokenizer.decode(token_id_j_disc[0] if token_id_j_disc.dim() > 1 else token_id_j_disc, skip_special_tokens=True).strip()
                    completion_i_gen_text = tokenizer.decode(token_id_i_gen[0] if token_id_i_gen.dim() > 1 else token_id_i_gen, skip_special_tokens=True).strip()
                    completion_j_gen_text = tokenizer.decode(token_id_j_gen[0] if token_id_j_gen.dim() > 1 else token_id_j_gen, skip_special_tokens=True).strip()
                    
                    pair_info_path = os.path.join(tracking_dir, f"{tracking_base_name}-step{global_step}-pair.csv")
                    with open(pair_info_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['item', 'disc_completion', 'gen_completion', 'val_score_after', 'gen_score_after', 'label', 'note'])
                        label_i = 'positive' if label_i.mean().item() > 0.5 else 'negative'
                        label_j = 'positive' if label_j.mean().item() > 0.5 else 'negative'
                        writer.writerow(['i (lower gold)', completion_i_disc_text, completion_i_gen_text, f'{score_i_disc.mean().item():.4f}', f'{score_i_gen.mean().item():.4f}', label_i, 'should be pushed DOWN'])
                        writer.writerow(['j (higher gold)', completion_j_disc_text, completion_j_gen_text, f'{score_j_disc.mean().item():.4f}', f'{score_j_gen.mean().item():.4f}', label_j, 'should be pushed UP'])
                    print(f"  Saved pair info to {pair_info_path}")
                
                # Clear cache to prevent memory accumulation
                torch.cuda.empty_cache()
            else:
                input_ids_i = batch["input_ids_i"].to(device)
                attention_mask_i = batch["attention_mask_i"].to(device)
                token_id_i = batch["token_id_i"].to(device)
                token_correct_i = batch["token_correct_i"].to(device)  # validator correct answer
                token_gen_i = batch["token_gen_i"].to(device)  # generator completion
                indicator_i = batch["indicator_i"].to(device)  # 1 if positive, 0 if negative

                input_ids_j = batch["input_ids_j"].to(device)
                attention_mask_j = batch["attention_mask_j"].to(device)
                token_id_j = batch["token_id_j"].to(device)
                token_correct_j = batch["token_correct_j"].to(device)  # validator correct answer
                token_gen_j = batch["token_gen_j"].to(device)  # generator completion
                indicator_j = batch["indicator_j"].to(device)  # 1 if positive, 0 if negative

                label = batch["label"].to(device)

                # Forward pass for prompt i
                outputs_i = model(input_ids=input_ids_i, attention_mask=attention_mask_i)
                # logits_i: [batch_size, seq_len, vocab_size]

                log_probs_i = F.log_softmax(outputs_i.logits, dim=-1)  # [B, seq_len, vocab_size]

                # Forward pass for prompt j
                outputs_j = model(input_ids=input_ids_j, attention_mask=attention_mask_j)
                log_probs_j = F.log_softmax(outputs_j.logits, dim=-1)  # [B, seq_len, vocab_size]
                
                # Compute scores - use log-odds for discriminator mode if flag is set
                if train_g_or_d == 'd' and validator_log_odds:
                    # Log-odds: log(sum P(yes_tokens)) - log(sum P(no_tokens))
                    def compute_logodds_simple(log_probs, token_ids):
                        """Compute log-odds for yes vs no at the position predicting the completion."""
                        batch_size = log_probs.shape[0]
                        logodds_list = []
                        for b in range(batch_size):
                            comp_len = token_ids[b].size(0)
                            pred_pos = -(comp_len + 1)
                            probs_at_pos = torch.exp(log_probs[b, pred_pos, :])
                            p_yes = probs_at_pos[yestoks].sum()
                            p_no = probs_at_pos[notoks].sum()
                            logodds = torch.log(p_yes + 1e-12) - torch.log(p_no + 1e-12)
                            logodds_list.append(logodds)
                        return torch.stack(logodds_list)
                    
                    score_i = compute_logodds_simple(log_probs_i, token_id_i)
                    score_j = compute_logodds_simple(log_probs_j, token_id_j)
                else:
                    # Log-probs (default) - apply length normalization for generator mode
                    use_lenorm = args.length_normalize and train_g_or_d == 'g'
                    score_i = sum_completion_logprobs(log_probs_i, token_id_i, length_normalize=use_lenorm)  # [B]
                    score_j = sum_completion_logprobs(log_probs_j, token_id_j, length_normalize=use_lenorm)  # [B]
                
                # Apply typicality correction to generator scores (online, during training)
                # Only for 'g' mode where score_i/score_j are generator scores
                if args.typicality_correction and train_g_or_d == 'g':
                    typicality_i = batch["typicality_i"].to(device)
                    typicality_j = batch["typicality_j"].to(device)
                    score_i = score_i - typicality_i
                    score_j = score_j - typicality_j
                
                # Apply validator boost (shift scores so optimal threshold is 0)
                # Only for 'd' mode where score_i/score_j are validator scores
                if args.boost_initial_val and train_g_or_d == 'd':
                    score_i = score_i + val_boost_theta
                    score_j = score_j + val_boost_theta

                # Use frozen reference model
                if WITH_REF:
                    # Check that token_ids are single-token (1D after squeeze, or 2D with size 1 in last dim)
                    # This reference model scoring code assumes single-token completions
                    if token_id_i.dim() > 1 or (token_id_i.dim() == 1 and token_id_i.size(0) != batch["input_ids_i"].size(0)):
                        raise NotImplementedError(
                            "Reference model scoring (WITH_REF) currently only supports single-token completions. "
                            f"Got token_id_i with shape {token_id_i.shape}. "
                            "Use --use-full-completion without --with_ref, or ensure completions are single tokens."
                        )
                    if token_id_j.dim() > 1 or (token_id_j.dim() == 1 and token_id_j.size(0) != batch["input_ids_j"].size(0)):
                        raise NotImplementedError(
                            "Reference model scoring (WITH_REF) currently only supports single-token completions. "
                            f"Got token_id_j with shape {token_id_j.shape}. "
                            "Use --use-full-completion without --with_ref, or ensure completions are single tokens."
                        )
                    
                    with torch.no_grad():
                        outputs_i_ref = model_ref(input_ids=input_ids_i, attention_mask=attention_mask_i, use_cache = False)
                        # logits_i: [batch_size, seq_len, vocab_size]
                        logits_i_ref = outputs_i_ref.logits
                        last_idx_i = attention_mask_i.size(1) - 1
                        selected_logits_i_ref = []
                        for b in range(logits_i_ref.size(0)):
                            selected_logits_i_ref.append(logits_i_ref[b, last_idx_i, :].unsqueeze(0))
                        selected_logits_i_ref = torch.cat(selected_logits_i_ref, dim=0)
                        log_probs_i_ref = F.log_softmax(selected_logits_i_ref, dim=-1)  # [B, vocab_size]
                        #score_i_ref = log_probs_i_ref[torch.arange(log_probs_i_ref.size(0)), token_id_i]
                        score_i_ref = log_probs_i_ref[torch.arange(log_probs_i_ref.size(0), device=device), token_id_i]

                        # Forward pass for prompt j
                        outputs_j_ref = model_ref(input_ids=input_ids_j, attention_mask=attention_mask_j, use_cache = False)
                        logits_j_ref = outputs_j_ref.logits

                        last_idx_j = attention_mask_j.size(1) - 1 # assumes LEFT padding
                        selected_logits_j_ref = []
                        for b in range(logits_j_ref.size(0)):
                            selected_logits_j_ref.append(logits_j_ref[b, last_idx_j, :].unsqueeze(0))
                        selected_logits_j_ref = torch.cat(selected_logits_j_ref, dim=0)
                        log_probs_j_ref = F.log_softmax(selected_logits_j_ref, dim=-1)  # [B, vocab_size]
                        #score_j_ref = log_probs_j_ref[torch.arange(log_probs_j_ref.size(0)), token_id_j]
                        score_j_ref = log_probs_j_ref[torch.arange(log_probs_j_ref.size(0), device=device), token_id_j]
                    diff_ref = score_j_ref - score_i_ref
                else:
                    diff_ref = 0

                # Pairwise logistic loss: - log( sigmoid( (score_j) - (score_i) ) )
                diff = score_j - score_i - diff_ref
                preference_loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
                
                # Validator NLL: -log P(correct_answer | prompt) for both items
                score_correct_i = sum_completion_logprobs(log_probs_i, token_correct_i)
                score_correct_j = sum_completion_logprobs(log_probs_j, token_correct_j)
                nll_validator_loss = -(score_correct_i + score_correct_j).mean() / 2
                
                # Generator NLL: -log P(completion | prompt) * indicator (only for positive examples)
                score_gen_i = sum_completion_logprobs(log_probs_i, token_gen_i)
                score_gen_j = sum_completion_logprobs(log_probs_j, token_gen_j)
                nll_generator_loss = -(score_gen_i * indicator_i + score_gen_j * indicator_j).mean() / 2
                
                # Total loss
                loss = preference_loss + nll_validator_weight * nll_validator_loss + nll_generator_weight * nll_generator_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                global_step += 1
                
                # Log to wandb
                if use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/preference_loss": preference_loss.item(),
                        "train/nll_validator_loss": nll_validator_loss.item(),
                        "train/nll_generator_loss": nll_generator_loss.item(),
                        "train/score_j": score_j.mean().item(),
                        "train/score_i": score_i.mean().item(),
                        "train/diff": diff.mean().item(),
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    if nll_validator_weight > 0:
                        log_dict["train/score_correct_i"] = score_correct_i.mean().item()
                        log_dict["train/score_correct_j"] = score_correct_j.mean().item()
                    if nll_generator_weight > 0:
                        log_dict["train/score_gen_i"] = score_gen_i.mean().item()
                        log_dict["train/score_gen_j"] = score_gen_j.mean().item()
                        log_dict["train/indicator_i"] = indicator_i.mean().item()
                        log_dict["train/indicator_j"] = indicator_j.mean().item()
                    wandb.log(log_dict)
                
                # Track scores for all datapoints at specified frequency
                if track_scores and global_step % track_scores_freq == 0:
                    print(f"\n  [Step {global_step}] Tracking scores for all datapoints...")
                    
                    # Get task_config for tracking
                    task_config_for_tracking = get_task(task)
                    
                    # Track all scores
                    tracked_results = track_all_scores(
                        model, tokenizer, L_train_all, task, device, yestoks, notoks,
                        length_normalize=args.length_normalize,
                        use_full_completion=use_full_completion,
                        task_config=task_config_for_tracking,
                        validator_log_odds=True,  # Always use log-odds for tracking (consistent with eval.py)
                        is_chat=with_chat,
                        has_system_role=has_system_role
                    )
                    
                    # Save to CSV
                    tracking_path = os.path.join(tracking_dir, f"{tracking_base_name}-step{global_step}.csv")
                    save_tracked_scores(tracked_results, tracking_path)
                    
                    # Also save info about the sampled pair
                    # Decode completions to get noun2 (the completion text)
                    completion_i_text = tokenizer.decode(token_id_i[0] if token_id_i.dim() > 1 else token_id_i, skip_special_tokens=True).strip()
                    completion_j_text = tokenizer.decode(token_id_j[0] if token_id_j.dim() > 1 else token_id_j, skip_special_tokens=True).strip()
                    gen_completion_i_text = tokenizer.decode(token_gen_i[0] if token_gen_i.dim() > 1 else token_gen_i, skip_special_tokens=True).strip()
                    gen_completion_j_text = tokenizer.decode(token_gen_j[0] if token_gen_j.dim() > 1 else token_gen_j, skip_special_tokens=True).strip()
                    
                    pair_info_path = os.path.join(tracking_dir, f"{tracking_base_name}-step{global_step}-pair.csv")
                    with open(pair_info_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['item', 'completion', 'gen_completion', 'score_after', 'label', 'note'])
                        # item i has LOWER gold score (should have lower score after training)
                        # item j has HIGHER gold score (should have higher score after training)
                        label_i = 'positive' if indicator_i.mean().item() > 0.5 else 'negative'
                        label_j = 'positive' if indicator_j.mean().item() > 0.5 else 'negative'
                        writer.writerow(['i (lower gold)', completion_i_text, gen_completion_i_text, f'{score_i.mean().item():.4f}', label_i, 'should be pushed DOWN'])
                        writer.writerow(['j (higher gold)', completion_j_text, gen_completion_j_text, f'{score_j.mean().item():.4f}', label_j, 'should be pushed UP'])
                    print(f"  Saved pair info to {pair_info_path}")
                
                # Clear cache to prevent memory accumulation
                torch.cuda.empty_cache()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if epoch%save_steps == 0 or epoch == num_epochs - 1:
            with_ref_str = "-with-ref" if with_ref else ""
            all_str = "-all" if use_all else ""

            if train_g_or_d == 'd':
                direction_str = '--g2d'
            elif train_g_or_d == 'g':
                direction_str = '--d2g'
            elif train_g_or_d == 'iter':
                direction_str = '--iter'
            elif train_g_or_d == 'both':
                direction_str = '--both'
            else:
                raise ValueError("not supported")

            split_type_str = "--"+ split_type

            alpha_str = "--alpha" + str(alpha) if isinstance(alpha, (int, float)) else "--alpha-" + str(alpha)
            typcorr_str = "--tc-online" if args.typicality_correction else ""  # tc = typicality correction, online = applied during training
            lenorm_str = "--lenorm" if args.length_normalize else ""
            single_token_str = "--single-token-data" if args.single_token_data_only else ""
            full_completion_str = "--full-completion" if use_full_completion else ""
            nll_v_str = f"--nllv{nll_validator_weight}" if nll_validator_weight > 0 else ""
            nll_g_str = f"--nllg{nll_generator_weight}" if nll_generator_weight > 0 else ""
            force_same_x_str = "--force-same-x" if args.force_same_x else ""
            valboost_str = "--valboost" if args.boost_initial_val else ""
            save_directory = "../models/v5-" + model_name.replace('/','--')  + "-delta"+str(delta)+"-epoch"+str(epoch) + "--" + task + with_ref_str + all_str + direction_str + split_type_str + alpha_str + typcorr_str + lenorm_str + single_token_str + full_completion_str + nll_v_str + nll_g_str + force_same_x_str + valboost_str
            print("Saving to ", save_directory)
            
            if use_lora:
                # For LoRA: Save adapters first, then merge and save full model
                print("Saving LoRA adapters...")
                model.save_pretrained(save_directory)
                
                print("Loading saved LoRA model...")
                from peft import AutoPeftModelForCausalLM
                model_peft = AutoPeftModelForCausalLM.from_pretrained(save_directory)
                
                print("Merging LoRA into base model...")
                merged_model = model_peft.merge_and_unload()
                
                merge_dir = save_directory + "_merged"
                print(f"Saving merged full model to {merge_dir}")
                merged_model.save_pretrained(merge_dir, safe_serialization=True, max_shard_size="2GB")
                tokenizer.save_pretrained(merge_dir)
                
                # Clean up merged model from memory
                del model_peft, merged_model
                torch.cuda.empty_cache()
            else:
                # For full model fine-tuning: Save normally
                model.save_pretrained(save_directory)
                tokenizer.save_pretrained(save_directory)
        
        # Log epoch-level metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/epoch": epoch + 1,
            })
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Weights & Biases run finished.")

if __name__ == "__main__":
    # Import tasks module to trigger registration of any custom tasks
    import tasks

    # Legacy task names (handled by existing if/elif chains)
    LEGACY_TASKS = ["hypernym", "hypernym-car", "trivia-qa", "swords", "lambada", "ifeval", "collie"]
    # Combined list includes both legacy and any newly registered tasks
    ALL_TASKS = get_all_task_names(LEGACY_TASKS)

    #TODO change --all flag since we can't set it to False like this!
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Model name/path")
    parser.add_argument("--task", type=str, choices=ALL_TASKS, help="Task to run")
    parser.add_argument("--with_ref", default=False, action="store_true", help="Whether to use reference model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--delta", type=float, default=10, help="Delta")
    parser.add_argument("--total_samples", type=int, default=5110, help="Total samples")
    parser.add_argument("--save_steps", type=int, default=1, help="Save steps")
    parser.add_argument("--all", default=True, action="store_true", help="Whether to use all examples or just positive ones")
    parser.add_argument("--train_g_or_d", type=str, default='d', choices=["d","g","iter","both"], help="Train generator or discriminator.")
    parser.add_argument("--split_type", type=str, default='random', choices=["random","hyper","both"], help="How to do train/test split. Only applies to hypernymy.")
    parser.add_argument("--alpha", type=str, default='1.0', help="Alpha value or function name. NOTE: this is only used when train_g_or_d is 'both'. Can be a number between 0 and 1, or 'alpha_fun_1'")
    parser.add_argument("--lora", action='store_true', help="Use LoRA for memory-efficient fine-tuning")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory (trades compute for memory)")
    parser.add_argument("--typicality-correction", action='store_true', help="Apply typicality correction: use (Generator - GPT-2 P(completion)) instead of raw Generator score")
    parser.add_argument("--no-full-completion", default=False, action='store_true', help="Use only first token for scoring instead of full completion (full completion is default)")
    parser.add_argument("--debug", action='store_true', help="Enable verbose debug output for tokenization checks")
    parser.add_argument("--single_token_data_only", action="store_true", default=False, help="Only use training data where generator completion is exactly one token")
    parser.add_argument("--nll_validator_weight", type=float, default=0.0, help="Weight for NLL loss on validator (discriminator) correct answers")
    parser.add_argument("--nll_generator_weight", type=float, default=0.0, help="Weight for NLL loss on generator completions (only for positive examples)")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name (auto-generated if not provided)")
    parser.add_argument("--no-v2", action="store_true", default=False, help="Use original hypernym data instead of v2 grammar-corrected data")
    parser.add_argument("--validator-log-odds", action="store_true", default=False, help="Use log-odds (log(P(Yes)/P(No))) for validator instead of log-probs (log(P(Yes)))")
    parser.add_argument("--length-normalize", action="store_true", default=False, help="Divide generator scores by number of tokens (length normalization)")
    parser.add_argument("--track-scores", action="store_true", default=False, help="Track gen/val scores for all datapoints during training")
    parser.add_argument("--track-scores-freq", type=int, default=10, help="Frequency (in steps) to track scores when --track-scores is enabled")
    parser.add_argument("--force-same-x", action="store_true", default=False, help="Only pair examples with the same generator prompt (same 'x'). Ensures pairs compare different completions for the same input.")
    parser.add_argument("--boost-initial-val", action="store_true", default=False, help="Shift validator scores so optimal classification threshold is 0. Computes theta = -optimal_threshold and adds it to all validator scores during training.")
    args = parser.parse_args()
    
    # Convert alpha to float if it's a number
    try:
        alpha_val = float(args.alpha)
        if 0 <= alpha_val <= 1:
            args.alpha = alpha_val
    except ValueError:
        if args.alpha not in ["alpha_fun_1"]:
            raise ValueError("Alpha must be a number between 0 and 1, or one of: alpha_fun_1")
    
    main(args)
