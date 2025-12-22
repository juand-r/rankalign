"""
This script is used to train a model to rank discriminator prompts to match the ranking of log-probabilities of generator prompts.

Usage:
python ranking_loss_ref.py --model google/gemma-2-2b --task hypernym --with_ref --num_epochs 10 --learning_rate 1e-5 --delta 5 --total_samples 5110 --save_steps 1

"""
import os
import sys
import itertools
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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
import utils
from utils import make_prompt_triviaqa, make_prompt_hypernymy, make_prompt_swords, make_prompt_lambada, make_prompt_ifeval, make_prompt_collie, get_final_logit_prob, get_completion_token_logprobs

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
    use_full_completion = args.use_full_completion
    debug = args.debug
    nll_validator_weight = args.nll_validator_weight
    nll_generator_weight = args.nll_generator_weight
    use_wandb = not args.no_wandb
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    WITH_REF = with_ref
    
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
                "single_token_only": args.single_token_only,
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

    if task=='hypernym':
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
    if args.single_token_only:
        print(f"Original L_train size: {len(L_train)}")
        # Determine the appropriate make_prompt function for the task
        if task in ['hypernym', 'hypernym-car']:
            make_prompt_fn = make_prompt_hypernymy
        elif task == 'trivia-qa':
            make_prompt_fn = make_prompt_triviaqa
        elif task == 'swords':
            make_prompt_fn = make_prompt_swords
        elif task == 'lambada':
            make_prompt_fn = make_prompt_lambada
        else:
            raise ValueError(f"Task {task} not supported for single_token_only filtering")
        
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


    if task in ['hypernym', 'hypernym-car']:
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
                    log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                    log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                    logprobs_last_layer.append((log_prob_d, log_prob_g))
                    #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
                else:
                    ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
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
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
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
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
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
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, both=None)
 
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
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
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

    # Apply typicality correction if requested
    if args.typicality_correction and train_g_or_d in ['d', 'both']:
        print("\n" + "="*60)
        print("APPLYING TYPICALITY CORRECTION")
        print("="*60)
        
        # Extract completions based on task
        completions = []
        if task in ['hypernym', 'hypernym-car']:
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
        
        # Apply correction to generator logprobs
        print("\nApplying correction: Generator - GPT-2 P(completion)")
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
            # For 'd' or 'g' mode, logprobs_last_layer is just a list of floats
            logprobs_original = logprobs_last_layer.copy()
            logprobs_last_layer = [lp - typicality_scores[i] for i, lp in enumerate(logprobs_last_layer)]
            original_mean = sum(logprobs_original) / len(logprobs_original)
            corrected_mean = sum(logprobs_last_layer) / len(logprobs_last_layer)
            print(f"  Original generator mean: {original_mean:.4f}")
            print(f"  Corrected generator mean: {corrected_mean:.4f}")
        
        print(f"  Correction applied to {len(logprobs_last_layer)} examples")
        
        # Clean up GPT-2 model
        del model_gpt2, tokenizer_gpt2
        torch.cuda.empty_cache()
        
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

    if train_g_or_d == 'both':
        # Create tuples of (discriminator_prompt, generator_prompt, logprobs)
        # Note: logprobs_last_layer contains tuples of (log_prob_d, log_prob_g)
        Z = list(zip(p_train_tune, p_train_gold, logprobs_last_layer))
        
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

        indices = range(len(Z))
        pair_inds = list(itertools.product(indices, repeat=2))
        pair_inds = [i for i in pair_inds if i[0] < i[1]]
        pair_inds = random.sample(pair_inds, total_samples)
        
        # Create pairs with all the information
        pairs_ = [(Z[i[0]], Z[i[1]]) for i in pair_inds]
    else:
        #Z = list(zip(prompts_pos, gen_logprobs_last_layer))
        # Include L_train_all to access ground truth labels (e.g., .taxonomic)
        Z = list(zip(p_train_tune, logprobs_last_layer, L_train_all))
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
        return tokenizer.decode(toks[1:])

    def get_correct_answer(data_item, task):
        """Get the ground truth answer (Yes/No) for a data item based on task type."""
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
            raise ValueError(f"Task {task} not supported for ground truth lookup")
        
        if label == 'yes':
            return space_prefix + "Yes"
        else:
            return space_prefix + "No"

    def get_generator_completion(data_item, task):
        """Get the generator completion (actual task answer) for a data item."""
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
                     (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task))  # indicators (1=positive, 0=negative)
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
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task))  # indicators (1=positive, 0=negative)
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
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task))  # indicators (1=positive, 0=negative)
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
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task))  # indicators (1=positive, 0=negative)
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
            pairs = [
                (
                    ((format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)), (completion_text, completion_text)),  # discriminator pair
                    ((format_with_inst(pair[0][1].prompt), format_with_inst(pair[1][1].prompt)), (pair[0][1].completion, pair[1][1].completion)),  # generator pair
                    (pair[0][0].completion.strip().lower()   , pair[1][0].completion.strip().lower()   )
                ) for pair in pairs_ if pair[1][-1][0] - pair[0][-1][0] > delta
            ]
        else:
            # Create pairs with both discriminator and generator prompts
            pairs = [
                (
                    ((pair[0][0].prompt, pair[1][0].prompt), (completion_text, completion_text)),  # discriminator pair
                    ((pair[0][1].prompt, pair[1][1].prompt), (pair[0][1].completion, pair[1][1].completion)),  # generator pair
                    (pair[0][0].completion.strip().lower()   , pair[1][0].completion.strip().lower()   )
                ) for pair in pairs_ if pair[1][-1][0] - pair[0][-1][0] > delta
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
                ((prompt_i_disc, prompt_j_disc), (completion_i_disc, completion_j_disc)), ((prompt_i_gen, prompt_j_gen), (completion_i_gen, completion_j_gen)), (label_i, label_j) = self.pairs[idx]
                if not self.use_full_completion:
                    completion_i_disc = self.tokenizer.decode(self.tokenizer.encode(completion_i_disc)[-1])
                    completion_j_disc = self.tokenizer.decode(self.tokenizer.encode(completion_j_disc)[-1])
                    completion_i_gen = self.tokenizer.decode(self.tokenizer.encode(completion_i_gen)[-1])
                    completion_j_gen = self.tokenizer.decode(self.tokenizer.encode(completion_j_gen)[-1])
            else:
                # Unified 5-element pair structure: (prompts, ranking_completions, validator_correct, gen_completions, indicators)
                (prompt_i, prompt_j), (completion_i, completion_j), (correct_i, correct_j), (gen_completion_i, gen_completion_j), (indicator_i, indicator_j) = self.pairs[idx]
                
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
                    'label': torch.tensor(1.0, dtype=torch.float)
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
                    'label_j': torch.tensor(1.0 if label_j == "yes" else 0.0, dtype=torch.float)
            }
            return item


    #18 fine for zero-shot
    if use_full_completion:
        batch_size = 1 #TODO: allow actual batches
    else:
        if with_ref:
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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            def sum_completion_logprobs(log_probs, token_ids):
                """
                log_probs: [batch, seq_len, vocab] - over input_ids (prompt + completion), left padded
                token_ids: [batch, completion_len] - only the completion tokens
                
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
                    completion_log_probs.append(gathered.sum())

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

                # Get scores
                score_i_disc = sum_completion_logprobs(log_probs_i_disc, token_id_i_disc)   
                score_j_disc = sum_completion_logprobs(log_probs_j_disc, token_id_j_disc)
                score_i_gen = sum_completion_logprobs(log_probs_i_gen, token_id_i_gen)
                score_j_gen = sum_completion_logprobs(log_probs_j_gen, token_id_j_gen)

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
                # Score for example i is the log-prob of token_id_i
                score_i = sum_completion_logprobs(log_probs_i, token_id_i)  # [B]

                # Forward pass for prompt j
                outputs_j = model(input_ids=input_ids_j, attention_mask=attention_mask_j)

                log_probs_j = F.log_softmax(outputs_j.logits, dim=-1)  # [B, seq_len, vocab_size]
                score_j = sum_completion_logprobs(log_probs_j, token_id_j)  # [B]

                # Use frozen reference model
                if WITH_REF:
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
            typcorr_str = "--typcorr" if args.typicality_correction else ""
            single_token_str = "--single-token" if args.single_token_only else ""
            full_completion_str = "--full-completion" if use_full_completion else ""
            nll_v_str = f"--nllv{nll_validator_weight}" if nll_validator_weight > 0 else ""
            nll_g_str = f"--nllg{nll_generator_weight}" if nll_generator_weight > 0 else ""
            save_directory = "../models/v5-" + model_name.replace('/','--')  + "-delta"+str(delta)+"-epoch"+str(epoch) + "--" + task + with_ref_str + all_str + direction_str + split_type_str + alpha_str + typcorr_str + single_token_str + full_completion_str + nll_v_str + nll_g_str
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Model name/path")
    parser.add_argument("--task", type=str, choices=["hypernym", "hypernym-car", "trivia-qa", "swords", "lambada", "ifeval", "collie"], help="Task to run")
    parser.add_argument("--with_ref", default=False, action="store_true", help="Whether to use reference model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--delta", type=float, default=10, help="Delta")
    parser.add_argument("--total_samples", type=int, default=5110, help="Total samples")
    parser.add_argument("--save_steps", type=int, default=1, help="Save steps")
    parser.add_argument("--all", default=False, action="store_true", help="Whether to use all examples or just positive ones")
    parser.add_argument("--train_g_or_d", type=str, default='d', choices=["d","g","iter","both"], help="Train generator or discriminator.")
    parser.add_argument("--split_type", type=str, default='random', choices=["random","hyper","both"], help="How to do train/test split. Only applies to hypernymy.")
    parser.add_argument("--alpha", type=str, default='1.0', help="Alpha value or function name. Can be a number between 0 and 1, or 'alpha_fun_1'")
    parser.add_argument("--lora", action='store_true', help="Use LoRA for memory-efficient fine-tuning")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory (trades compute for memory)")
    parser.add_argument("--typicality-correction", action='store_true', help="Apply typicality correction: use (Generator - GPT-2 P(completion)) instead of raw Generator score")
    parser.add_argument("--use-full-completion", default=False, action='store_true', help="Use full completion for generator scoring instead of just the first token")
    parser.add_argument("--debug", action='store_true', help="Enable verbose debug output for tokenization checks")
    parser.add_argument("--single_token_only", action="store_true", default=False, help="Only use training data where generator completion is exactly one token")
    parser.add_argument("--nll_validator_weight", type=float, default=0.0, help="Weight for NLL loss on validator (discriminator) correct answers")
    parser.add_argument("--nll_generator_weight", type=float, default=0.0, help="Weight for NLL loss on generator completions (only for positive examples)")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable Weights & Biases logging (enabled by default)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name (auto-generated if not provided)")
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
