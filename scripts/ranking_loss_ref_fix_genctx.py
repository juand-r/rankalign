"""
FIX1 g-mode training fork of scripts/ranking_loss_ref.py.

This script is intentionally narrower than the parent script:

  1. Only --train_g_or_d g is supported. d / both / iter raise.
  2. Only registry tasks are supported. Legacy task branches raise.
  3. Only full-completion, batch_size=1 training is supported.
  4. with_ref, track_scores, single-token-only, and non-g modes raise.

Main fix1 behavior:

  1. Pair construction uses the 4-shape consistent pool:
       case_A     = (L_neg, L_pos)
       mixed_neg  = (L_neg, U)
       mixed_pos  = (U,     L_pos)
       both_U     = (U,     U)
     Pairs with labeled items on the wrong natural side are dropped.

  2. Sampling is per-prompt x per-shape with within-prompt backfill.
     Shape weights are controlled by --shape-weight-* flags.

  3. Generator NLL fires per item, only for labeled positives:
       weight = is_labeled_item * indicator_item

  4. Validator NLL fires per item for every labeled item:
       weight = is_labeled_item

  5. Validator NLL position is fixed for g-mode: when val-NLL is on,
     the dataset builds discriminator-side inputs `disc_prompt + " Yes"`,
     the train loop runs a second discriminator forward pass, and Yes/No
     log-odds are read at the discriminator answer slot rather than inside
     the generator statement.

  6. Legacy pair_is_labeled AND-gating is not used by the active loss.

  7. Trained checkpoints use the v7- prefix and --fix1 suffix.

Known caveats:

  - --include-eos is not safe with non-log-odds val-NLL until the
    discriminator-side tail is also made to include EOS.
  - --batch-size > 1 is disabled; variable-length token fields still need
    a custom collate function before batching can be safely re-enabled.
  - --force-same-x behavior still needs a separate cleanup/toggle pass.

See docs/issue3_fix.md and docs/comb_loss_g_mode_concerns.md for rationale.
Use scripts/ranking_loss_ref.py for parent d/both/iter behavior.
"""
import os
import sys
import subprocess
import itertools
import csv
from pathlib import Path
from collections import defaultdict
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import math
import re
import random
import argparse
import wandb
import numpy as np
from datetime import datetime
import json

from datasets import load_dataset
from sklearn.metrics import roc_curve

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
import utils
#from utils import make_prompt_triviaqa, make_prompt_hypernymy, make_prompt_swords, make_prompt_lambada, make_prompt_ifeval, make_prompt_collie, 
from utils import get_final_logit_prob, get_completion_token_logprobs

from task_registry import get_task, get_all_task_names


def _chat_template_input_ids(enc):
    """Unwrap apply_chat_template output (tensor or BatchEncoding) for transformers >=5.

    Older versions and some tokenizers (Gemma) return a raw tensor when
    return_tensors='pt'; newer versions and some tokenizers (Qwen) return a
    BatchEncoding (dict-like with .input_ids). This wrapper makes downstream
    `.shape[1]` / subscript access work for both cases.
    """
    if hasattr(enc, "input_ids"):
        return enc.input_ids
    if isinstance(enc, dict):
        return enc["input_ids"]
    return enc


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


def compute_self_typicality_training(completions, model, tokenizer, device,
                                     is_chat=False, has_system_role=False, include_eos=False,
                                     disable_thinking=False, debug=False):
    """
    Compute self-typicality: unconditional log P_model(completion) using the
    scoring model itself (instead of GPT-2).

    For each completion, computes log P(completion | null_context) where
    null_context is BOS (base models) or a chat-formatted empty prompt
    (instruction-tuned models).
    """
    typicality_scores = []

    print("\nComputing self-typicality scores (using scoring model itself)...")
    with torch.no_grad():
        for idx, completion in enumerate(tqdm(completions, desc="Self typicality")):
            token_logprobs = get_completion_token_logprobs(
                "", completion, model, tokenizer, device,
                is_chat=is_chat, has_system_role=has_system_role,
                include_eos=include_eos,
                disable_thinking=disable_thinking,
            )
            if debug and idx < 3:
                print(f"[DEBUG self-typ] idx={idx} completion={completion!r}")
                print(f"[DEBUG self-typ] token_logprobs={token_logprobs.tolist()} sum={float(token_logprobs.sum().item()):.6f}")
            typicality_scores.append(float(token_logprobs.sum().item()))

    print(f"  Computed {len(typicality_scores)} self-typicality scores")
    if len(typicality_scores) > 0:
        print(f"  Mean self-typicality: {sum(typicality_scores)/len(typicality_scores):.4f}")

    return typicality_scores


def compute_neg_typicality_training(L_train_all, task, make_prompt_fn,
                                    model, tokenizer, device,
                                    is_chat=False, has_system_role=False, include_eos=False,
                                    disable_thinking=False, debug=False):
    """Compute log P(completion | negated_prompt) for each training item.

    Uses make_negated_gen_prompt from eval_by_claude.py to construct the
    negated prompts, ensuring train/eval consistency.
    """
    from eval_by_claude import make_negated_gen_prompt

    neg_scores = []

    print("\nComputing neg-typicality scores (negated-prompt LLR denominator)...")
    with torch.no_grad():
        for idx, item in enumerate(tqdm(L_train_all, desc="Neg typicality")):
            neg_prompt, completion = make_negated_gen_prompt(item, task, make_prompt_fn)
            token_logprobs = get_completion_token_logprobs(
                neg_prompt, completion, model, tokenizer, device,
                is_chat=is_chat, has_system_role=has_system_role,
                include_eos=include_eos,
                disable_thinking=disable_thinking,
            )
            if debug and idx < 3:
                print(f"[DEBUG neg-typ] idx={idx} neg_prompt={neg_prompt!r}")
                print(f"[DEBUG neg-typ] completion={completion!r}")
                print(f"[DEBUG neg-typ] token_logprobs={token_logprobs.tolist()} sum={float(token_logprobs.sum().item()):.6f}")
            neg_scores.append(float(token_logprobs.sum().item()))

    print(f"  Computed {len(neg_scores)} neg-typicality scores")
    if len(neg_scores) > 0:
        print(f"  Mean neg-typicality: {sum(neg_scores)/len(neg_scores):.4f}")

    return neg_scores


def split_prompts_labeled_unlabeled(prompts, ratio, seed):
    """Split unique prompts into labeled/unlabeled sets.
    
    Args:
        prompts: list of prompt strings (one per training item, may repeat)
        ratio: fraction of unique prompts to mark as labeled
        seed: random seed for reproducibility
    
    Returns:
        labeled_set: set of prompt strings that are labeled
    """
    unique_prompts = sorted(set(prompts))
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)
    n_labeled = max(1, int(len(unique_prompts) * ratio))
    labeled_set = set(unique_prompts[:n_labeled])
    return labeled_set


def get_tracking_base_filename(model_name, task, delta, train_g_or_d, use_all, split_type, alpha,
                                typicality_correction, length_normalize, use_full_completion,
                                preference_loss_weight, nll_validator_weight, nll_generator_weight,
                                force_same_x=False, boost_initial_val=False,
                                self_typicality=False, neg_typicality=False,
                                semi_supervised=None, labeled_only=None):
    """Generate base filename for tracking logs (same as model save name but without epoch)."""
    direction_str = {'d': 'g2d', 'g': 'd2g', 'iter': 'iter', 'both': 'both'}[train_g_or_d]
    all_str = "-all" if use_all else ""
    alpha_str = f"-alpha{alpha}" if isinstance(alpha, (int, float)) else f"-alpha-{alpha}"
    if neg_typicality:
        typcorr_str = "-tc-neg"
    elif self_typicality:
        typcorr_str = "-tc-self"
    elif typicality_correction:
        typcorr_str = "-tc-online"
    else:
        typcorr_str = ""
    lenorm_str = "-lenorm" if length_normalize else ""
    full_completion_str = "-full-completion" if use_full_completion else ""
    pref_str = f"-pref{preference_loss_weight}" if preference_loss_weight != 1.0 else ""
    nll_v_str = f"-nllv{nll_validator_weight}" if nll_validator_weight > 0 else ""
    nll_g_str = f"-nllg{nll_generator_weight}" if nll_generator_weight > 0 else ""
    force_same_x_str = "-force-same-x" if force_same_x else ""
    valboost_str = "-valboost" if boost_initial_val else ""
    if semi_supervised is not None:
        semi_str = f"-semi{semi_supervised}"
    elif labeled_only is not None:
        semi_str = f"-labelonly{labeled_only}"
    else:
        semi_str = ""
    
    base_name = (f"v5-{model_name.replace('/', '--')}-delta{delta}--{task}{all_str}"
                 f"--{direction_str}--{split_type}{alpha_str}{typcorr_str}{lenorm_str}"
                 f"{full_completion_str}{pref_str}{nll_v_str}{nll_g_str}{force_same_x_str}{valboost_str}{semi_str}"
                 f"--fix1")
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
    # ranking_loss_ref_fix.py: this fork supports ONLY g-mode training. The
    # per-item gen-NLL framing and the 4-shape consistent-pair pool are
    # g-mode-specific (the loss block edits only cover g-mode). See
    # docs/comb_loss_g_mode_concerns.md for the full rationale.
    if train_g_or_d != 'g':
        raise NotImplementedError(
            f"ranking_loss_ref_fix.py only supports --train_g_or_d g "
            f"(got {train_g_or_d!r}). Use scripts/ranking_loss_ref.py for d/both/iter modes."
        )
    split_type = args.split_type
    alpha = args.alpha  # New alpha parameter
    use_lora = args.lora
    gradient_checkpointing = args.gradient_checkpointing
    use_full_completion = not args.no_full_completion
    debug = args.debug
    preference_loss_weight = args.preference_loss_weight
    nll_validator_weight = args.nll_validator_weight
    nll_generator_weight = args.nll_generator_weight
    use_wandb = not args.no_wandb
    validator_log_odds = args.validator_log_odds
    track_scores = args.track_scores
    track_scores_freq = args.track_scores_freq
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup tracking directory and base filename
    if track_scores:
        raise NotImplementedError("tracking scores is not supported in fix1")

    WITH_REF = with_ref
    
    # Yes/No token variants for log-odds computation
    yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
    no_words = ["No", " No", "NO", "no", " no"]
    
    # Initialize wandb if enabled
    if use_wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            # Auto-generate run name from key parameters
            pref_str = f"-pref{preference_loss_weight}" if preference_loss_weight != 1.0 else ""
            if args.semi_supervised is not None:
                semi_str = f"-semi{args.semi_supervised}"
            elif args.labeled_only is not None:
                semi_str = f"-labelonly{args.labeled_only}"
            else:
                semi_str = ""
            model_short = model_name.split('/')[-1]
            bins_str = f"-bins{args.delta_bins}" if args.delta_bins is not None else ""
            run_name = f"{model_short}-{task}-{train_g_or_d}-delta{delta}{bins_str}-nllv{nll_validator_weight}-nllg{nll_generator_weight}{pref_str}{semi_str}-lr{lr}"
        
        wandb.init(
            project="rankalign",
            name=run_name,
            config={
                "model": model_name,
                "task": task,
                "train_g_or_d": train_g_or_d,
                "delta": delta,
                "preference_loss_weight": preference_loss_weight,
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
                "typicality_correction": args.typicality_correction,
                "self_typicality": args.self_typicality,
                "neg_typicality": args.neg_typicality,
                "semi_supervised": args.semi_supervised,
                "labeled_only": args.labeled_only,
                "split_seed": args.split_seed,
                "consistency_ft": bool(args.consistency_ft),
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

    _name_looks_instruct = (
        'Instruct' in model_name or 'instruct' in model_name or '-it' in model_name
    )
    # Qwen3/3.5 post-trained models omit "Instruct" from the name (e.g. Qwen3-4B).
    # Their base models are explicitly named with "-Base" (e.g. Qwen3-4B-Base).
    _qwen3_post_trained = (
        re.search(r'[Qq]wen3', model_name) is not None and 'Base' not in model_name
    )
    if _name_looks_instruct or _qwen3_post_trained:
        with_chat = True
        if _name_looks_instruct:
            print(f"Detected instruct model (name match): {model_name}")
        else:
            print(f"Detected instruct model (Qwen3+ post-trained): {model_name}")
        print("Using chat template formatting for prompts")
        disc_shots = "zero"
        space_prefix = ""
    else:
        with_chat = False
        disc_shots = "few"
        space_prefix = " "
        print(f"Using standard formatting for model: {model_name}")

    if args.disc_shots is not None:
        disc_shots = args.disc_shots
        print(f"Overriding disc_shots to: {disc_shots}")

    has_system_role = False
    if 'llama' in model_name.lower() or 'qwen' in model_name.lower():
        has_system_role = True
        print("Model has system role!")

    # For Qwen3+ post-trained models, disable hybrid reasoning mode in chat template.
    # We don't evaluate reasoning models, so we want enable_thinking=False threaded
    # through every apply_chat_template call. For non-Qwen3+ models, chat_template_kwargs
    # is empty and behavior is byte-identical to before.
    disable_thinking = _qwen3_post_trained
    chat_template_kwargs = {"enable_thinking": False} if disable_thinking else {}
    if disable_thinking:
        print("Qwen3+ post-trained: disabling thinking mode in chat template (enable_thinking=False)")

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
    def dbg(msg):
        if debug:
            print(f"[DEBUG] {msg}")
    
    # Compute yes/no token IDs for log-odds computation
    yestoks = [tokenizer.encode(w)[-1] for w in yes_words]
    notoks = [tokenizer.encode(w)[-1] for w in no_words]
    if validator_log_odds:
        print(f"Using log-odds for validator: yestoks={yestoks}, notoks={notoks}")

        # Fail-fast: the log-odds path (both pair selection and val-NLL) assumes
        #   (a) the disc tail "{space_prefix}Yes" tokenizes to exactly 1 token,
        #       so compute_logodds_simple's pred_pos = -(comp_len + 1) = -2 lines
        #       up with the answer slot in `disc_prompt + tail`, AND
        #   (b) every variant in yes_words / no_words tokenizes to exactly 1
        #       token, so yestoks / notoks (built via tokenizer.encode(w)[-1])
        #       capture the model's full "say yes" / "say no" probability mass
        #       at the answer slot.
        # Both hold for Gemma-2 ("Yes"=3553, " Yes"=6287). For other tokenizers
        # (e.g. some Qwen variants) a multi-token tail would silently mis-index
        # the position lookup, and multi-token variants in yes_words/no_words
        # would silently undercount aggregated mass (only the last sub-token
        # would be in yestoks/notoks). Catch both at startup, before any GPU
        # work, with an actionable error.
        disc_yes_tail = space_prefix + "Yes"
        disc_yes_ids = tokenizer.encode(disc_yes_tail, add_special_tokens=False)
        if len(disc_yes_ids) != 1:
            raise ValueError(
                f"--validator-log-odds requires the disc tail {disc_yes_tail!r} "
                f"to tokenize to exactly 1 token (got {len(disc_yes_ids)}: "
                f"{disc_yes_ids} -> {[tokenizer.decode([t]) for t in disc_yes_ids]!r}). "
                f"compute_logodds_simple uses pred_pos = -(comp_len + 1) and "
                f"assumes comp_len == 1; a multi-token tail would read the wrong "
                f"position. Generalize compute_logodds_simple to be robust to "
                f"multi-token tails (read at the slot predicting the FIRST tail "
                f"token and aggregate over yes/no variants there), OR run without "
                f"--validator-log-odds (the non-log-odds val-NLL path uses "
                f"sum_completion_logprobs and is already robust to multi-token "
                f"completions)."
            )
        for word_list, name in [(yes_words, "yes_words"), (no_words, "no_words")]:
            for w in word_list:
                ids = tokenizer.encode(w, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(
                        f"--validator-log-odds requires every variant in {name} "
                        f"to tokenize to exactly 1 token. {w!r} -> {ids} "
                        f"({[tokenizer.decode([t]) for t in ids]!r}). yestoks / "
                        f"notoks are built via tokenizer.encode(w)[-1] which "
                        f"silently drops the prefix sub-tokens of multi-token "
                        f"variants, so probs[yestoks].sum() at the answer slot "
                        f"misses mass that the model actually puts on this "
                        f"variant. Either (1) drop {w!r} from {name} above (line "
                        f"~287-288) so we only aggregate single-token variants, "
                        f"or (2) generalize the aggregation to handle multi-"
                        f"token variants (e.g. compute log P(variant) by "
                        f"summing logprobs over its full token sequence)."
                    )
    if debug:
        dbg(f"with_chat={with_chat} has_system_role={has_system_role} space_prefix={space_prefix!r} disc_shots={disc_shots}")
        dbg(f"yes token variants: {[(w, tokenizer.encode(w, add_special_tokens=False), tokenizer.decode([tokenizer.encode(w)[-1]])) for w in yes_words]}")
        dbg(f"no token variants: {[(w, tokenizer.encode(w, add_special_tokens=False), tokenizer.decode([tokenizer.encode(w)[-1]])) for w in no_words]}")
    
    # Conditionally add LoRA for memory-efficient fine-tuning
    if use_lora:
        print("Setting up LoRA for memory-efficient fine-tuning...")
        if args.gemma4_lora:
            # Gemma 4 wraps each projection in Gemma4ClippableLinear, so the inner
            # Linear lives at <projection>.linear and PEFT cannot find it by exact
            # module name. The regex matches paths ending in q_proj/k_proj/.../down_proj
            # for both Gemma 4 (paths ending in .linear under a projection) and Gemma 2
            # (where the projection IS the Linear). Vision-tower modules are excluded
            # because Gemma 4 is multimodal.
            target_modules = r"^(?!.*vision_tower).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
            print("  [--gemma4-lora] using regex target_modules (excludes vision_tower)")
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Llama target modules
        lora_config = LoraConfig(
            r=16,  # Low-rank dimension
            lora_alpha=32,  # LoRA scaling parameter
            target_modules=target_modules,
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
    # LoRA + gradient_checkpointing breaks grad_fn unless inputs are forced to require grads.
    # PEFT >=0.15 has enable_input_require_grads(); older versions (e.g. 0.14 on gemma-2 pods)
    # don't — use a forward hook on the embedding layer instead, which works for any version.
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(
                lambda module, inp, out: out.requires_grad_(True)
            )
        print("Gradient checkpointing enabled")
    else:
        print("Gradient checkpointing disabled")

    if WITH_REF:
        raise NotImplementedError("with_ref is not supported in fix1")
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
    else:
        raise NotImplementedError("Task not implemented!")

    # Filter for single-token completions if requested
    if args.single_token_data_only:
        raise NotImplementedError("single_token_data_only is not supported in fix1")
    # Drop items exceeding --max-seq-len before they hit the dataloader
    # (which would truncate input_ids but not completion token_ids, silently
    # misaligning scores in sum_completion_logprobs and risking a gather crash).
    # Measure the exact (style, shots, completion) combinations training will
    # tokenize: 'g' uses generator/zero, 'd' uses discriminator/disc_shots+" Yes",
    # 'both' tokenizes both sequences so take the max.

    if args.max_seq_len and args.max_seq_len > 0 and task_config is not None:
        before = len(L_train)

        def _encode_len(prompt_text, completion_text):
            if with_chat:
                msgs = (
                    [{"role": "system", "content": "You are a helpful assistant."}]
                    if has_system_role else []
                ) + [{"role": "user", "content": prompt_text}]
                n_prompt = len(_chat_template_input_ids(tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors='pt', **chat_template_kwargs))[0])
                n_completion = len(tokenizer.encode(completion_text, add_special_tokens=False))
                return n_prompt + n_completion
            return len(tokenizer.encode(prompt_text + completion_text, add_special_tokens=False))

        def _fits(item):
            lengths = []
            if train_g_or_d in ('g', 'both'):
                pc = task_config['make_prompt'](item, style='generator', shots='zero')
                lengths.append(_encode_len(pc.prompt, pc.completion))
            # g-mode only needs train-time discriminator inputs when val-NLL is on.
            if train_g_or_d in ('d', 'both') or (train_g_or_d == 'g' and nll_validator_weight > 0):
                pc = task_config['make_prompt'](item, style='discriminator', shots=disc_shots)
                lengths.append(_encode_len(pc.prompt, space_prefix + "Yes"))
            if not lengths:
                pc = task_config['make_prompt'](item, style='generator', shots='zero')
                lengths.append(_encode_len(pc.prompt, pc.completion))
            return max(lengths) <= args.max_seq_len

        L_train = [item for item in L_train if _fits(item)]
        if len(L_train) < before:
            print(f"[max-seq-len filter] Dropped {before - len(L_train)}/{before} items "
                  f"exceeding {args.max_seq_len} tokens")

    # --- TAIL-MISMATCH DROP FILTER (this copy only; NOT in shared ranking_loss_ref_fix.py) ---
    # Some prompt/completion pairs tokenize so that the completion's tokens differ when
    # appended to the prompt vs standalone (BPE merge at the prompt->completion seam; the v2
    # humaneval format glues the body onto the signature colon with no separator). The
    # trainer's _check_tail guard aborts on those. qwen's tokenizer merges that seam ~6% of
    # the time (gemma ~0.3%). Per the user's decision: DROP those items before training
    # rather than train on misaligned completion positions. Faithful to __getitem__: raw
    # tokenize(prompt+completion) vs tokenize(completion); also checks the disc " Yes" tail
    # when val-NLL is on (s4). copy-only; the shared trainer stays byte-identical.
    if task_config is not None:
        def _tail_ok(_prompt, _completion):
            _comp = tokenizer.encode(_completion, add_special_tokens=False)
            if len(_comp) == 0:
                return True
            _full = tokenizer(_prompt + _completion)["input_ids"]
            return _full[-len(_comp):] == _comp
        def _item_ok(_item):
            _gp = task_config['make_prompt'](_item, style='generator', shots='zero')
            if not _tail_ok(_gp.prompt, _gp.completion):
                return False
            if nll_validator_weight > 0:
                _dp = task_config['make_prompt'](_item, style='discriminator', shots=disc_shots)
                if not _tail_ok(_dp.prompt, space_prefix + "Yes"):
                    return False
            return True
        _before_tm = len(L_train)
        L_train = [it for it in L_train if _item_ok(it)]
        _dropped_tm = _before_tm - len(L_train)
        print(f"[tail-mismatch filter] Dropped {_dropped_tm}/{_before_tm} items "
              f"(prompt/completion BPE seam merge -> _check_tail would abort). copy-only.")
    # --- end TAIL-MISMATCH DROP FILTER ---

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

        # Compute validator scores for pair selection.
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            if validator_log_odds:
                probs = get_final_logit_prob(
                    prompt, model, tokenizer, device,
                    is_chat=with_chat, has_system_role=has_system_role,
                    disable_thinking=disable_thinking,
                )
                p_yes = probs[yestoks].sum()
                p_no = probs[notoks].sum()
                score = float((torch.log(p_yes + 1e-12) - torch.log(p_no + 1e-12)).item())
                if debug and idx < 3:
                    dbg(f"pair-score idx={idx} style={gold_prompt_style} metric=validator_logodds")
                    dbg(f"pair-score prompt={prompt[:300]!r}")
                    dbg(f"pair-score p_yes={float(p_yes.item()):.6f} p_no={float(p_no.item()):.6f} logodds={score:.6f}")
            else:
                target_text = space_prefix + "Yes"
                log_prob = get_completion_token_logprobs(
                    prompt, target_text, model, tokenizer, device,
                    is_chat=with_chat, has_system_role=has_system_role,
                    disable_thinking=disable_thinking,
                )
                score = float(log_prob.sum().item())
                if debug and idx < 3:
                    dbg(f"pair-score idx={idx} style={gold_prompt_style} target={target_text!r}")
                    dbg(f"pair-score prompt={prompt[:300]!r}")
                    dbg(f"pair-score target_ids={tokenizer.encode(target_text, add_special_tokens=False)} token_logprobs={log_prob.tolist()} sum={score:.6f}")

            logprobs_last_layer.append(score)

        # Generate tune prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(
            task_config['make_prompt'], L_train_all, tokenizer,
            style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None
        )
        if debug:
            dbg(f"p_train_gold[0]: prompt={p_train_gold[0].prompt[:300]!r} completion={p_train_gold[0].completion!r}")
            dbg(f"p_train_tune[0]: prompt={p_train_tune[0].prompt[:300]!r} completion={p_train_tune[0].completion!r}")
            dbg(f"logprobs_last_layer[0:3]={logprobs_last_layer[:3]}")

    else:
        raise NotImplementedError("Legacy tasks no longer supported in fix1")

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
            completions = [task_config['get_completion'](item) for item in L_train_all]
        else:
            raise NotImplementedError("Legacy tasks no longer supported in fix1")
        
        if args.neg_typicality:
            # Neg-typicality: log P(completion | negated_prompt)
            print("\nUsing NEG-TYPICALITY (negated-prompt LLR)")
            if task_config is not None:
                make_prompt_fn = task_config['make_prompt']
            else:
                raise NotImplementedError("Legacy tasks no longer supported in fix1")

            typicality_scores = compute_neg_typicality_training(
                L_train_all, task, make_prompt_fn,
                model, tokenizer, device,
                is_chat=with_chat, has_system_role=has_system_role,
                include_eos=args.include_eos,
                disable_thinking=disable_thinking,
                debug=debug,
            )
        elif args.self_typicality:
            # Self-typicality: use the scoring model itself
            print("\nUsing SELF-TYPICALITY (scoring model as its own prior)")
            typicality_scores = compute_self_typicality_training(
                completions, model, tokenizer, device,
                is_chat=with_chat, has_system_role=has_system_role,
                include_eos=args.include_eos,
                disable_thinking=disable_thinking,
                debug=debug,
            )
        else:
            raise NotImplementedError("GPT-2 typicality is not wired in fix1; use --self-typicality or --neg-typicality")

        print(f"  Computed typicality scores for {len(typicality_scores)} examples")
        print(f"  Typicality mean: {sum(typicality_scores)/len(typicality_scores):.4f}")

        print("="*60 + "\n")

    if with_chat and has_system_role:
        # Process discriminator prompts (p_train_tune)
        # Use "assistant" role (Gemma maps it to "model" internally; Qwen/Llama use it natively)
        ms_tune = [ [ {"role": "system", "content": "You are a helpful assistant."},  {"role": "user", "content": i.prompt.strip()}, {"role": "assistant", "content": i.completion.strip()} ] for i in p_train_tune]
        toks_tune = tokenizer.apply_chat_template(ms_tune, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt', **chat_template_kwargs)
        max_context_length = _chat_template_input_ids(toks_tune).shape[1]
        
        if train_g_or_d in ('both',) or (train_g_or_d == 'g' and nll_validator_weight > 0):
            ms_gold = [ [ {"role": "system", "content": "You are a helpful assistant."},  {"role": "user", "content": i.prompt.strip()}, {"role": "assistant", "content": i.completion.strip()} ] for i in p_train_gold]
            toks_gold = tokenizer.apply_chat_template(ms_gold, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt', **chat_template_kwargs)
            max_context_length = max(max_context_length, _chat_template_input_ids(toks_gold).shape[1])
    elif with_chat:
        # Process discriminator prompts (p_train_tune)
        ms_tune = [ [ {"role": "user", "content": i.prompt.strip()}, {"role": "assistant", "content": i.completion.strip()} ] for i in p_train_tune]
        toks_tune = tokenizer.apply_chat_template(ms_tune, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt', **chat_template_kwargs)
        max_context_length = _chat_template_input_ids(toks_tune).shape[1]
        
        if train_g_or_d in ('both',) or (train_g_or_d == 'g' and nll_validator_weight > 0):
            ms_gold = [ [ {"role": "user", "content": i.prompt.strip()}, {"role": "assistant", "content": i.completion.strip()} ] for i in p_train_gold]
            toks_gold = tokenizer.apply_chat_template(ms_gold, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt', **chat_template_kwargs)
            max_context_length = max(max_context_length, _chat_template_input_ids(toks_gold).shape[1])
    else:
        #TODO later should make this cleaner in utils.make_and_format_data
        max_context_length = len(hf_train[0]['input_ids'])
        if train_g_or_d in ('both',) or (train_g_or_d == 'g' and nll_validator_weight > 0):
            max_context_length = max(len(hf_train_gold[0]['input_ids']), max_context_length)
    print("MAX CONTEXT LENGTH: ", max_context_length)
    if args.max_seq_len is not None and args.max_seq_len > 0:
        if max_context_length > args.max_seq_len:
            print(
                f"Capping max_context_length {max_context_length} -> {args.max_seq_len} "
                f"(via --max-seq-len)"
            )
            max_context_length = args.max_seq_len

    # Prepare typicality scores for inclusion in Z (use zeros if not computed)
    typ_scores_for_z = typicality_scores if typicality_scores is not None else [0.0] * len(L_train_all)

    # --- Semi-supervised / labeled-only prompt split ---
    is_labeled_flags = [True] * len(L_train_all)
    if args.semi_supervised is not None or args.labeled_only is not None:
        ratio = args.semi_supervised if args.semi_supervised is not None else args.labeled_only
        all_prompts = [pt.prompt for pt in p_train_tune]
        labeled_set = split_prompts_labeled_unlabeled(all_prompts, ratio, args.split_seed)
        is_labeled_flags = [pt.prompt in labeled_set for pt in p_train_tune]

        n_labeled = sum(is_labeled_flags)
        n_unlabeled = len(is_labeled_flags) - n_labeled
        unique_labeled = len(labeled_set)
        unique_total = len(set(all_prompts))
        mode_name = "semi-supervised" if args.semi_supervised is not None else "labeled-only"
        print(f"\n{'='*60}")
        print(f"PROMPT SPLIT ({mode_name}, ratio={ratio}, seed={args.split_seed})")
        print(f"{'='*60}")
        print(f"Unique prompts: {unique_total} total, {unique_labeled} labeled, {unique_total - unique_labeled} unlabeled")
        print(f"Items: {len(is_labeled_flags)} total, {n_labeled} labeled, {n_unlabeled} unlabeled")

        if args.labeled_only is not None:
            keep = [i for i, flag in enumerate(is_labeled_flags) if flag]
            L_train_all = [L_train_all[i] for i in keep]
            p_train_tune = [p_train_tune[i] for i in keep]
            p_train_gold = [p_train_gold[i] for i in keep]
            logprobs_last_layer = [logprobs_last_layer[i] for i in keep]
            typ_scores_for_z = [typ_scores_for_z[i] for i in keep]
            is_labeled_flags = [True] * len(L_train_all)
            print(f"Filtered to {len(L_train_all)} labeled items")
        print(f"{'='*60}\n")

    # --- Consistency-FT filter (opt-in, SFT-only) ---
    # Computes per-item generator scores (raw log P(completion|gen_prompt))
    # in a single forward-only pass, derives mean thresholds t_v, t_g over
    # validator/generator scores, binarizes each item, and DROPS items whose
    # binarized labels disagree. Pass-through (no behavior change) when
    # --consistency-ft is not set. Argparse already enforces:
    #   pref_w == 0, nll_v_w > 0, nll_g_w > 0, force_same_x == False.
    consistency_ft_stats = None
    if args.consistency_ft:
        print("\n" + "="*60)
        print("CONSISTENCY-FT FILTER (validator/generator agreement)")
        print("="*60)
        print(f"Computing per-item generator scores for {len(p_train_tune)} items "
              f"(raw log P(completion|gen_prompt), no length-norm, no typicality)...")
        gen_scores = []
        for idx, pc in enumerate(tqdm(p_train_tune)):
            log_prob = get_completion_token_logprobs(
                pc.prompt, pc.completion, model, tokenizer, device,
                is_chat=with_chat, has_system_role=has_system_role,
                disable_thinking=disable_thinking,
            )
            gen_scores.append(float(log_prob.sum().item()))
            if debug and idx < 3:
                dbg(f"cft idx={idx} prompt={pc.prompt[:200]!r}")
                dbg(f"cft completion={pc.completion!r} gen_score={gen_scores[-1]:.6f}")

        val_scores = list(logprobs_last_layer)

        # Compute thresholds. With --semi-supervised, base means on labeled
        # items only (so t_v, t_g reflect the supervised distribution; the
        # filter then only consults bv/bg for labeled items anyway).
        if args.semi_supervised is not None:
            ref_idx = [i for i, lf in enumerate(is_labeled_flags) if lf]
            if not ref_idx:
                raise ValueError("--consistency-ft: no labeled items to compute thresholds")
            v_for_mean = [val_scores[i] for i in ref_idx]
            g_for_mean = [gen_scores[i] for i in ref_idx]
            t_basis = "labeled-only"
        else:
            if not val_scores:
                raise ValueError("--consistency-ft: no items to compute thresholds")
            v_for_mean = val_scores
            g_for_mean = gen_scores
            t_basis = "all-items"
        t_v = sum(v_for_mean) / len(v_for_mean)
        t_g = sum(g_for_mean) / len(g_for_mean)

        bv = [1 if v > t_v else 0 for v in val_scores]
        bg = [1 if g > t_g else 0 for g in gen_scores]

        cell_counts_labeled = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for i in range(len(L_train_all)):
            if is_labeled_flags[i]:
                cell_counts_labeled[(bv[i], bg[i])] += 1

        # Filter: drop labeled items where bv != bg; keep all unlabeled.
        keep = []
        n_unlabeled_kept = 0
        for i in range(len(L_train_all)):
            if not is_labeled_flags[i]:
                keep.append(i)
                n_unlabeled_kept += 1
            elif bv[i] == bg[i]:
                keep.append(i)

        n_labeled_total = sum(is_labeled_flags)
        n_labeled_kept = (cell_counts_labeled[(0, 0)] +
                          cell_counts_labeled[(1, 1)])
        n_labeled_dropped = n_labeled_total - n_labeled_kept

        L_train_all = [L_train_all[i] for i in keep]
        p_train_tune = [p_train_tune[i] for i in keep]
        p_train_gold = [p_train_gold[i] for i in keep]
        logprobs_last_layer = [logprobs_last_layer[i] for i in keep]
        typ_scores_for_z = [typ_scores_for_z[i] for i in keep]
        is_labeled_flags = [is_labeled_flags[i] for i in keep]

        consistency_ft_stats = {
            "t_v": float(t_v),
            "t_g": float(t_g),
            "threshold_basis": t_basis,
            "n_total_pre_filter": int(len(bv)),
            "n_labeled_total": int(n_labeled_total),
            "n_labeled_kept": int(n_labeled_kept),
            "n_labeled_dropped": int(n_labeled_dropped),
            "n_unlabeled_kept": int(n_unlabeled_kept),
            "n_kept": int(len(L_train_all)),
            "label_cells_bv_bg": {
                "00_low_low":   int(cell_counts_labeled[(0, 0)]),
                "01_low_high":  int(cell_counts_labeled[(0, 1)]),
                "10_high_low":  int(cell_counts_labeled[(1, 0)]),
                "11_high_high": int(cell_counts_labeled[(1, 1)]),
            },
        }
        print(f"Threshold basis: {t_basis}")
        print(f"  t_v (mean validator score) = {t_v:.4f}")
        print(f"  t_g (mean generator score) = {t_g:.4f}")
        print(f"Labeled-item binarization cells (bv, bg):")
        print(f"  (0,0) low-low   = {cell_counts_labeled[(0, 0)]:>5}  KEPT")
        print(f"  (1,1) high-high = {cell_counts_labeled[(1, 1)]:>5}  KEPT")
        print(f"  (0,1) low-high  = {cell_counts_labeled[(0, 1)]:>5}  dropped")
        print(f"  (1,0) high-low  = {cell_counts_labeled[(1, 0)]:>5}  dropped")
        print(f"Kept {len(L_train_all)} items: "
              f"{n_labeled_kept} labeled-agree + {n_unlabeled_kept} unlabeled "
              f"(dropped {n_labeled_dropped} disagreement-labeled)")
        print(f"{'='*60}\n")

    if train_g_or_d == 'both':
        raise NotImplementedError("both mode is not supported in fix1")
    else:
        # FIX1: g-mode pair construction with the 4-shape consistent-pair pool.
        # See docs/comb_loss_g_mode_concerns.md "Locked design" section.
        # Each item is partitioned by gold label (L+ / L- / U). A pair (i, j)
        # with val(i) < val(j) is "consistent" iff every labeled item is on
        # its natural side: L+ on HI (j), L- on LO (i). That collapses into
        # 4 shapes:
        #   case_A     = L_neg x L_pos     (both labeled, consistent)
        #   mixed_neg  = L_neg x U         (labeled neg on lo, unlabeled on hi)
        #   mixed_pos  = U     x L_pos     (unlabeled on lo, labeled pos on hi)
        #   both_U     = U     x U         (no labels, validator decides)
        # Pairs from any other shape (L+/L+, L-/L-, L+/L-, L+/U-on-lo,
        # U-on-hi/L-) are dropped.
        #
        # --force-same-x composes with the 4-shape filter: when fsx is on, the
        # partition into L+/L-/U is done WITHIN each prompt group, and pairs
        # never cross prompts (parent fsx semantics preserved).

        # FIX1 (val-NLL position fix, 2026-05-22): Z now also carries
        # p_train_gold (= discriminator prompts in g-mode) as element 5.
        # Used in the dataset to tokenize a 2nd input sequence
        # `disc_prompt + " Yes"` so the val-NLL term reads log-odds at
        # the actual answer slot, not at a position inside the generator
        # statement. See concern #1 in docs/comb_loss_g_mode_concerns.md.
        # (p_train_tune, validator_score, L_train_all, typicality, is_labeled, p_train_gold)
        Z = list(zip(p_train_tune, logprobs_last_layer, L_train_all, typ_scores_for_z, is_labeled_flags, p_train_gold))
        Z = sorted(Z, key=lambda i: i[1])  # sort ascending by validator score

        min_logprob = Z[0][1]
        max_logprob = Z[-1][1]
        score_name = "validator log-odds" if validator_log_odds else "validator logprob"

        # Auto-delta (fix1, 2026-05-22): if --delta-bins N is set, override
        # --delta with (p95-p5)/N of the validator-score distribution. This
        # rescales delta to whatever score metric is in use (log-odds vs
        # log-prob) and to the model/task spread. p5-p95 (not min-max) keeps
        # outliers from blowing up the spread. See "Delta calibration" in
        # docs/comb_loss_g_mode_concerns.md.
        scores_arr = np.asarray([z[1] for z in Z], dtype=float)
        p5_score, p95_score = np.percentile(scores_arr, [5, 95])
        spread_5_95 = float(p95_score - p5_score)
        delta_auto = None
        if args.delta_bins is not None:
            if args.delta_bins <= 0:
                raise ValueError(f"--delta-bins must be > 0, got {args.delta_bins}")
            delta_auto = spread_5_95 / args.delta_bins
            print(f"Auto-delta: spread(p5-p95)={spread_5_95:.4f}, "
                  f"bins={args.delta_bins}, score_metric={score_name}")
            print(f"Auto-delta: overriding --delta {delta} -> {delta_auto:.4f}")
            delta = float(delta_auto)
        else:
            print(f"Delta (minimum separation, fixed): {delta}  "
                  f"(spread(p5-p95)={spread_5_95:.4f}, score_metric={score_name})")
        if delta != 0:
            NN = (max_logprob - min_logprob) / delta
            print(f"NN (full min-max range / delta): {NN}")
        print(f"Min {score_name}: {min_logprob}")
        print(f"Max {score_name}: {max_logprob}")

        # Append auto-delta info to a log file in the models directory so we
        # can audit/compare across runs after-the-fact.
        try:
            os.makedirs(args.models_dir, exist_ok=True)
            log_path = os.path.join(args.models_dir, "auto_delta_log.csv")
            log_exists = os.path.exists(log_path)
            with open(log_path, "a") as f:
                if not log_exists:
                    f.write("timestamp,model,task,score_metric,n_items,"
                            "min,p5,p95,max,spread_p5_p95,delta_bins,"
                            "delta_used,delta_arg\n")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts},{model_name},{task},{score_name},"
                        f"{len(scores_arr)},{min_logprob:.6f},"
                        f"{p5_score:.6f},{p95_score:.6f},{max_logprob:.6f},"
                        f"{spread_5_95:.6f},{args.delta_bins},"
                        f"{delta:.6f},{args.delta:.6f}\n")
            print(f"Auto-delta log appended to {log_path}")
        except Exception as e:
            print(f"[warn] could not write auto-delta log: {e}")

        # Update wandb config with the post-override delta and spread so the
        # dashboard reflects what was actually used (wandb.init ran before
        # auto-delta override).
        if use_wandb and wandb.run is not None:
            try:
                wandb.config.update(
                    {
                        "delta": delta,
                        "delta_bins": args.delta_bins,
                        "validator_score_spread_p5_p95": spread_5_95,
                        "validator_score_min": min_logprob,
                        "validator_score_max": max_logprob,
                    },
                    allow_val_change=True,
                )
            except Exception as e:
                print(f"[warn] wandb.config.update failed: {e}")

        # Per-item label lookup (used here for pair-shape partitioning; the
        # downstream pairs[] builder uses get_indicator with the same logic).
        _task_config = get_task(task)

        def _label_class_for_z(z_tuple):
            is_lab = z_tuple[4]
            if not is_lab:
                return 'U'
            data_item = z_tuple[2]

            if _task_config is not None:
                ind = _task_config['get_indicator'](data_item)
            else:
                raise NotImplementedError("Legacy tasks no longer supported in fix1")
            return 'L_pos' if ind >= 0.5 else 'L_neg'

        #TODO later make this more efficient
        def _enumerate_shape_subset(lo_pool, hi_pool, delta_local):
            """All (i, j) with i in lo_pool, j in hi_pool, i != j,
            val(i) < val(j), |val(j) - val(i)| > delta_local.

            delta_local: scalar threshold for THIS subset (global delta when
            --per-prompt-delta is off; per-prompt delta when it is on).

            TODO(perf): this is O(|lo_pool| * |hi_pool|) in time AND memory
            (the surviving pairs are materialized into a Python list). For the
            both_U shape in semi-supervised g-mode this is O(|U|^2), which is
            ~1.8M iterations / ~80MB for persona-v1 (n=1500) and starts to hurt
            for n > ~10k. Same complexity profile as the parent
            ranking_loss_ref.py non-fsx branch (pre-existing, not a fix1
            regression). When this becomes a bottleneck, replace with
            rejection sampling: draw random (i in lo_pool, j in hi_pool),
            check (val_i < val_j) + delta_local, retry on miss, until we have
            total_samples_for_this_shape pairs. That's O(N) memory and
            O(N / acceptance_rate) time."""
            pairs = []
            for i in lo_pool:
                v_i = Z[i][1]
                for j in hi_pool:
                    if i == j:
                        continue
                    v_j = Z[j][1]
                    if v_i < v_j and (v_j - v_i) > delta_local:
                        pairs.append((i, j))
            return pairs

        print(f"\n{'='*60}")
        print(f"FIX1 G-MODE PAIR CONSTRUCTION (per-prompt x per-shape)")
        print(f"  force_same_x={args.force_same_x}  per_prompt_delta={args.per_prompt_delta}")
        print(f"{'='*60}")
        # See docs/issue3_fix.md for the full algorithm + rationale.

        # 1. Group indices by prompt. When fsx is OFF the entire dataset is
        #    treated as one "virtual prompt" (key=None). This unifies the two
        #    code paths so per-prompt allocation logic runs identically.
        if args.force_same_x:
            prompt_to_indices = defaultdict(list)
            for idx, z in enumerate(Z):
                prompt_to_indices[z[0].prompt].append(idx)
            prompt_groups = dict(prompt_to_indices)
        else:
            prompt_groups = {None: list(range(len(Z)))}
        print(f"Prompts: {len(prompt_groups)} group(s) "
              f"({'fsx on' if args.force_same_x else 'fsx off -> 1 virtual group'})")

        # 2. Per-prompt x per-shape enumeration. When --per-prompt-delta is on,
        #    also compute a local delta per prompt from that prompt's score
        #    subset. Otherwise every prompt uses the global delta.
        pool_by_prompt: dict = {}       # pool_by_prompt[prompt][shape] -> list of (i, j)
        prompt_n: dict = {}              # prompt_n[prompt] -> num completions in this group
        delta_by_prompt: dict = {}       # delta_by_prompt[prompt] -> delta_local
        per_prompt_score_stats: dict = {}  # prompt -> {min,p5,p95,max,spread,delta_local}
        n_lpos_total = n_lneg_total = n_u_total = 0
        for prompt, indices in prompt_groups.items():
            # Per-prompt delta. Falls back to global delta when not requested
            # OR when the prompt has too few items for percentiles to be
            # meaningful (np.percentile works for n>=1 but is uninformative;
            # we keep the global delta in that case for safety).
            local_scores = np.asarray([Z[k][1] for k in indices], dtype=float)
            if local_scores.size >= 2:
                p5_loc, p95_loc = np.percentile(local_scores, [5, 95])
            else:
                p5_loc = p95_loc = float(local_scores[0]) if local_scores.size == 1 else 0.0
            spread_loc = float(p95_loc - p5_loc)
            if args.per_prompt_delta and args.delta_bins is not None and local_scores.size >= 2:
                delta_loc = spread_loc / args.delta_bins
            else:
                delta_loc = delta  # fall back to the global delta
            delta_by_prompt[prompt] = delta_loc
            per_prompt_score_stats[prompt] = {
                "n_items": int(local_scores.size),
                "min": float(local_scores.min()) if local_scores.size else 0.0,
                "p5": float(p5_loc),
                "p95": float(p95_loc),
                "max": float(local_scores.max()) if local_scores.size else 0.0,
                "spread_p5_p95": spread_loc,
                "delta_used": float(delta_loc),
            }

            grp_lpos, grp_lneg, grp_u = [], [], []
            for k in indices:
                cls = _label_class_for_z(Z[k])
                if cls == 'L_pos':
                    grp_lpos.append(k)
                elif cls == 'L_neg':
                    grp_lneg.append(k)
                else:
                    grp_u.append(k)
            n_lpos_total += len(grp_lpos)
            n_lneg_total += len(grp_lneg)
            n_u_total += len(grp_u)
            pool_by_prompt[prompt] = {
                'case_A':    _enumerate_shape_subset(grp_lneg, grp_lpos, delta_loc),
                'mixed_neg': _enumerate_shape_subset(grp_lneg, grp_u,    delta_loc),
                'mixed_pos': _enumerate_shape_subset(grp_u,    grp_lpos, delta_loc),
                'both_U':    _enumerate_shape_subset(grp_u,    grp_u,    delta_loc),
            }
            prompt_n[prompt] = len(indices)

        print(f"|L+| = {n_lpos_total}  |L-| = {n_lneg_total}  |U| = {n_u_total}  "
              f"(of {len(Z)} total)")

        # Per-prompt delta diagnostic (only print when actually using per-prompt
        # delta -- otherwise every prompt has the same global delta, no info).
        if args.per_prompt_delta and len(prompt_groups) > 1:
            local_deltas = [s["delta_used"] for s in per_prompt_score_stats.values()]
            local_spreads = [s["spread_p5_p95"] for s in per_prompt_score_stats.values()]
            print(f"Per-prompt delta: min={min(local_deltas):.4f}, "
                  f"median={float(np.median(local_deltas)):.4f}, "
                  f"max={max(local_deltas):.4f}  "
                  f"(global delta would be {delta:.4f})")
            print(f"Per-prompt spread(p5-p95): min={min(local_spreads):.4f}, "
                  f"median={float(np.median(local_spreads)):.4f}, "
                  f"max={max(local_spreads):.4f}  "
                  f"(global spread is {spread_5_95:.4f})")

        # Aggregate diagnostics: per-shape totals across all prompts.
        agg_pool = {s: 0 for s in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U')}
        for prompt in pool_by_prompt:
            for s in agg_pool:
                agg_pool[s] += len(pool_by_prompt[prompt][s])
        for s in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U'):
            print(f"  {s:10s}: {agg_pool[s]:8d} valid pairs (after delta filter)")
        total_valid_pairs = sum(agg_pool.values())
        print(f"  total     : {total_valid_pairs:8d}")
        if total_valid_pairs == 0:
            raise ValueError(
                f"No valid pairs after delta + consistency filters (delta={delta}). "
                f"Pool sizes: {agg_pool}. Try reducing --delta or check label distribution."
            )

        # 3. Per-prompt budget allocation, proportional to completion count
        #    (every completion gets equal expected exposure to training).
        N_total = len(Z)
        if total_samples > total_valid_pairs:
            print(f"\nWARNING: Reducing total_samples from {total_samples} to {total_valid_pairs} "
                  f"(not enough valid pairs across all prompts/shapes)")
            total_samples = total_valid_pairs

        shape_weights = {
            'case_A':    args.shape_weight_case_a,
            'mixed_neg': args.shape_weight_mixed_neg,
            'mixed_pos': args.shape_weight_mixed_pos,
            'both_U':    args.shape_weight_both_u,
        }
        sw_sum = sum(shape_weights.values())
        if sw_sum <= 0:
            raise ValueError(f"All shape weights are zero or negative: {shape_weights}")

        # 4. Sample pairs. Two modes:
        #    - "per-prompt" (DEFAULT, original behavior): per-prompt budget
        #      proportional to n_p, then within-prompt shape-stratified
        #      sampling with within-prompt backfill. Under fsx + prompt-level
        #      labeling, each prompt only has one nonzero shape, so the
        #      global case_A/both_U mix is determined by the labeled-prompt
        #      fraction (NOT by shape weights). See docs/issue3_fix.md.
        #    - "global" (OPT-IN): per-shape global budget first (weights
        #      renormalized over nonzero-pool shapes; deficit redistributed
        #      to non-saturated shapes); then per-prompt allocation within
        #      each shape proportional to that prompt's pool of that shape.
        #      With fsx + prompt-level labeling this restores meaningful
        #      shape-weight control of the case_A vs both_U mix and uses
        #      ALL labeled (case_A) pairs every epoch instead of ~42%.
        #      See chat 2026-05-24.
        pair_inds = []
        sampled_per_shape = {s: 0 for s in shape_weights}
        per_prompt_log: list = []  # (prompt, n_p, prompt_budget, n_sampled) for diagnostics
        per_shape_budget: dict = {}  # shape -> int (only populated in "global" mode)

        if args.shape_budget_mode == "per-prompt":
            for prompt in pool_by_prompt:
                n_p = prompt_n[prompt]
                prompt_budget = int(round(total_samples * n_p / N_total))
                prompt_pool = pool_by_prompt[prompt]
                prompt_sampled: list = []

                # First pass: shape-stratified sampling within this prompt.
                for shape, w in shape_weights.items():
                    target = int(round(prompt_budget * w / sw_sum))
                    avail = prompt_pool[shape]
                    take = min(target, len(avail))
                    if take > 0:
                        picked = random.sample(avail, take)
                        prompt_sampled.extend(picked)
                        sampled_per_shape[shape] += take

                # Within-prompt backfill: any deficit relative to prompt_budget is
                # filled from this prompt's remaining pool, weighted uniformly by
                # leftover size (cross-shape within the prompt).
                deficit = prompt_budget - len(prompt_sampled)
                if deficit > 0:
                    already = set(prompt_sampled)
                    leftover = []  # (shape, pair_tuple)
                    for shape, pool in prompt_pool.items():
                        for p in pool:
                            if p not in already:
                                leftover.append((shape, p))
                    if leftover:
                        fill = random.sample(leftover, min(deficit, len(leftover)))
                        for shape, p in fill:
                            prompt_sampled.append(p)
                            sampled_per_shape[shape] += 1

                pair_inds.extend(prompt_sampled)
                per_prompt_log.append((prompt, n_p, prompt_budget, len(prompt_sampled)))

        elif args.shape_budget_mode == "global":
            # Step A: global per-shape budget. Renormalize weights over the
            # shapes that actually have a nonzero pool, so the user-supplied
            # weights behave like ratios in the regime where some shapes are
            # structurally zero (e.g. fsx + prompt-level labeling -> only
            # case_A and both_U have pairs).
            nonzero_shapes = [s for s in shape_weights if agg_pool[s] > 0]
            if not nonzero_shapes:
                raise ValueError(
                    "shape-budget-mode=global: no nonzero-pool shapes "
                    "(should never happen, the upstream guard caught this)"
                )
            ew = {s: shape_weights[s] for s in nonzero_shapes}
            ew_sum = sum(ew.values())
            for s in nonzero_shapes:
                per_shape_budget[s] = int(round(total_samples * ew[s] / ew_sum))

            # Step B: cap each shape's budget by its pool, redistribute any
            # resulting deficit proportionally to the non-saturated shapes.
            # Loop in case redistribution itself saturates more shapes (rare
            # but possible). Bounded by len(nonzero_shapes) iterations.
            for _ in range(len(nonzero_shapes) + 1):
                deficit = 0
                for s in nonzero_shapes:
                    if per_shape_budget[s] > agg_pool[s]:
                        deficit += per_shape_budget[s] - agg_pool[s]
                        per_shape_budget[s] = agg_pool[s]
                if deficit == 0:
                    break
                non_saturated = [s for s in nonzero_shapes
                                 if per_shape_budget[s] < agg_pool[s]]
                if not non_saturated:
                    break  # everything is at pool cap; total < total_samples
                ns_w_sum = sum(ew[s] for s in non_saturated)
                for s in non_saturated:
                    add = int(round(deficit * ew[s] / ns_w_sum))
                    per_shape_budget[s] = min(per_shape_budget[s] + add,
                                              agg_pool[s])

            # Step C: within each shape, allocate per-prompt budget proportional
            # to that prompt's pool of that shape, then sample. This preserves
            # "every (prompt,shape) pair gets exposure proportional to its pool
            # size" within a shape, while the cross-shape ratio is now driven
            # by the user's weights rather than the labeled-fraction.
            samples_per_prompt_run: dict = defaultdict(int)
            for shape in nonzero_shapes:
                budget_s = per_shape_budget[shape]
                prompts_with_shape = [p for p in pool_by_prompt
                                      if len(pool_by_prompt[p][shape]) > 0]
                if not prompts_with_shape or budget_s == 0:
                    continue
                shape_pool_total = agg_pool[shape]
                taken: list = []
                for p in prompts_with_shape:
                    pool_p = pool_by_prompt[p][shape]
                    target = int(round(budget_s * len(pool_p) / shape_pool_total))
                    target = min(target, len(pool_p))
                    if target > 0:
                        picked = random.sample(pool_p, target)
                        taken.extend((p, q) for q in picked)
                # Backfill any rounding loss within this shape (typically a few
                # pairs lost to int(round(...)) accumulation). Sample uniformly
                # from the union of remaining pairs across prompts of this shape.
                deficit_within = budget_s - len(taken)
                if deficit_within > 0:
                    already = {q for (_p, q) in taken}
                    leftover = []
                    for p in prompts_with_shape:
                        for q in pool_by_prompt[p][shape]:
                            if q not in already:
                                leftover.append((p, q))
                    if leftover:
                        fill = random.sample(leftover,
                                             min(deficit_within, len(leftover)))
                        taken.extend(fill)
                # Commit this shape's picks.
                for (p, q) in taken:
                    pair_inds.append(q)
                    sampled_per_shape[shape] += 1
                    samples_per_prompt_run[p] += 1

            # Build per_prompt_log so the existing diagnostic prints + JSON-log
            # block work unchanged. prompt_budget=None signals "global mode";
            # n_sampled is the actual count.
            for prompt in pool_by_prompt:
                per_prompt_log.append((prompt, prompt_n[prompt], None,
                                       samples_per_prompt_run[prompt]))

        else:
            raise ValueError(
                f"Unknown --shape-budget-mode: {args.shape_budget_mode!r} "
                f"(must be 'per-prompt' or 'global')"
            )

        random.shuffle(pair_inds)

        # Diagnostics.
        print(f"\nSampled per shape (aggregated across prompts):")
        for shape in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U'):
            n = sampled_per_shape[shape]
            avail = agg_pool[shape]
            w = shape_weights[shape]
            print(f"  {shape:10s}: {n:6d}/{avail:8d}  (weight={w})")
        print(f"Total sampled: {len(pair_inds)}/{total_samples}")

        # Global-mode per-shape budget diagnostic.
        if args.shape_budget_mode == "global":
            print(f"\nGlobal-mode per-shape budget (after redistribution):")
            for s in ('case_A', 'mixed_neg', 'mixed_pos', 'both_U'):
                bud = per_shape_budget.get(s, 0)
                print(f"  {s:10s}: budget={bud:6d}  pool={agg_pool[s]:8d}  "
                      f"weight={shape_weights[s]}")

        # Per-prompt summary (only print details if there are >1 prompts).
        if len(prompt_groups) > 1:
            sorted_log = sorted(per_prompt_log, key=lambda r: r[3], reverse=True)
            print(f"\nPer-prompt sampling (top 3 by sample count, then bottom 3):")
            def _fmt(r):
                p, n_p, b, s = r
                p_str = (p[:60] + '...') if isinstance(p, str) and len(p) > 60 else str(p)
                budget_str = "n/a" if b is None else f"{b:5d}"
                return f"  n_p={n_p:5d}  budget={budget_str}  sampled={s:5d}  prompt={p_str!r}"
            for r in sorted_log[:3]:
                print(_fmt(r))
            if len(sorted_log) > 6:
                print(f"  ... ({len(sorted_log) - 6} prompts in middle) ...")
            for r in sorted_log[-3:]:
                print(_fmt(r))

        # Debug: show 3 sample pairs.
        print(f"\n--- Sample pairs (first 3) ---")
        for pi, (i, j) in enumerate(pair_inds[:3]):
            ci = _label_class_for_z(Z[i])
            cj = _label_class_for_z(Z[j])
            print(f"Pair {pi+1}: ({ci} -> {cj})")
            print(f"  Prompt i: '{Z[i][0].prompt[:80]}...'")
            print(f"  Compl  i: '{Z[i][0].completion}' (logprob={Z[i][1]:.3f})")
            print(f"  Compl  j: '{Z[j][0].completion}' (logprob={Z[j][1]:.3f})")
        print(f"{'='*60}\n")

        # Per-run JSON log: a single self-contained record of the pair-
        # construction stats for this training run. Includes global score
        # spread, the global delta, and (when fsx is on) per-prompt stats
        # with per-prompt deltas. Written under <models-dir>/training_run_logs/.
        try:
            samples_per_prompt: dict = {p: s for (p, _n, _b, s) in per_prompt_log}
            valid_pairs_per_prompt: dict = {
                p: {sh: len(pool_by_prompt[p][sh]) for sh in pool_by_prompt[p]}
                for p in pool_by_prompt
            }
            run_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name,
                "task": task,
                "flags": {
                    "force_same_x": bool(args.force_same_x),
                    "per_prompt_delta": bool(args.per_prompt_delta),
                    "shape_budget_mode": args.shape_budget_mode,
                    "delta_bins": args.delta_bins,
                    "delta_arg": float(args.delta),
                    "validator_log_odds": bool(validator_log_odds),
                    "self_typicality": bool(args.self_typicality),
                    "neg_typicality": bool(args.neg_typicality),
                    "semi_supervised": args.semi_supervised,
                    "labeled_only": args.labeled_only,
                    "split_seed": args.split_seed,
                    "consistency_ft": bool(args.consistency_ft),
                },
                "consistency_ft": consistency_ft_stats,
                "shape_weights": dict(shape_weights),
                "per_shape_budget": dict(per_shape_budget),  # {} when mode='per-prompt'
                "score_metric": score_name,
                "n_items_total": len(Z),
                "global_score_stats": {
                    "min": float(min_logprob),
                    "p5": float(p5_score),
                    "p95": float(p95_score),
                    "max": float(max_logprob),
                    "spread_p5_p95": float(spread_5_95),
                    "delta_used": float(delta),
                },
                "label_partition": {
                    "L_pos": int(n_lpos_total),
                    "L_neg": int(n_lneg_total),
                    "U":     int(n_u_total),
                },
                "valid_pairs_by_shape": dict(agg_pool),
                "sampled_by_shape":     dict(sampled_per_shape),
                "total_valid_pairs": int(total_valid_pairs),
                "total_sampled": int(len(pair_inds)),
                "n_prompt_groups": len(prompt_groups),
            }
            # Per-prompt detail (only meaningful when fsx is on; we still
            # emit the single None-keyed entry when fsx is off, for symmetry).
            per_prompt_records = []
            for p, stats in per_prompt_score_stats.items():
                rec = dict(stats)  # n_items, min, p5, p95, max, spread_p5_p95, delta_used
                rec["prompt"] = p if p is not None else "<all-prompts-fsx-off>"
                rec["valid_pairs_by_shape"] = valid_pairs_per_prompt.get(p, {})
                rec["n_sampled"] = int(samples_per_prompt.get(p, 0))
                per_prompt_records.append(rec)
            # Sort by n_items desc for readability.
            per_prompt_records.sort(key=lambda r: -r["n_items"])
            run_log["per_prompt"] = per_prompt_records

            log_root = os.path.join(args.models_dir, "training_run_logs")
            os.makedirs(log_root, exist_ok=True)
            ts_safe = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short_safe = model_name.replace("/", "--")
            log_name = f"{ts_safe}_{model_short_safe}_{task}.json"
            log_path = os.path.join(log_root, log_name)
            with open(log_path, "w") as f:
                json.dump(run_log, f, indent=2, default=str)
            print(f"Per-run training log written to {log_path}")
        except Exception as e:
            print(f"[warn] could not write per-run training log: {e}")

        pairs_ = [(Z[i[0]], Z[i[1]]) for i in pair_inds]


    def format_with_inst(prompt):
        if has_system_role:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},]
        else:
            message = [
                {"role": "user", "content": prompt},]
        toks = _chat_template_input_ids(tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors='pt', **chat_template_kwargs))[0]
        # Strip leading BOS if present (Gemma/Llama prepend BOS; Qwen does not)
        has_leading_bos = (
            tokenizer.bos_token_id is not None
            and len(toks) > 0
            and toks[0].item() == tokenizer.bos_token_id
        )
        toks_content = toks[1:] if has_leading_bos else toks
        decoded = tokenizer.decode(toks_content)
        
        # Assert: decode/re-encode should produce the same tokens
        # If this fails, there's a tokenization asymmetry that could cause training inconsistencies
        reencoded = tokenizer.encode(decoded, add_special_tokens=False, return_tensors='pt')[0]
        assert torch.equal(reencoded, toks_content), (
            f"Decode/re-encode mismatch! "
            f"Original tokens (no BOS): {toks_content.tolist()}, "
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
        else:
            raise NotImplementedError("Legacy tasks no longer supported in fix1")
       
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

    def get_indicator(data_item, task):
        """Get indicator (1 if positive example, 0 if negative)."""
        # Check task registry first (for new extensible tasks)
        task_config = get_task(task)
        if task_config is not None:
            # NEW PATH: Use registered task configuration
            return task_config['get_indicator'](data_item)
        else:
            raise NotImplementedError("Legacy tasks no longer supported in fix1")
        
        return 1.0 if label == 'yes' else 0.0


    if train_g_or_d=='d':
        raise NotImplementedError("d mode is not supported in fix1")
    elif train_g_or_d=='g':
        #NOTE in this case the ranking is derived from the log-probs of Yes under both prompts but we are targetting
        # the log-odds (hopefully log-prob is fine here) of the *generator completion*, so not the same in each item of the pair!
        # FIX1 (val-NLL position fix, 2026-05-22): pair tuple now also carries
        # the **discriminator** prompt (8th element). The dataset uses it to
        # tokenize a 2nd input `disc_prompt + " Yes"` for val-NLL.
        # Note: the delta filter is NOT re-applied here. Pairs in pairs_ were
        # already filtered by _enumerate_shape_subset using the appropriate
        # delta (global, or per-prompt under --per-prompt-delta). Re-applying
        # a single global delta here would over-filter the per-prompt-delta
        # case.
        if with_chat:
            pairs = [
                (
                    (format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)),  # prompts
                    (pair[0][0].completion, pair[1][0].completion),  # completion for ranking (generator completions)
                    (get_correct_answer(pair[0][2], task), get_correct_answer(pair[1][2], task)),  # validator correct answers
                    (pair[0][0].completion, pair[1][0].completion),  # generator completions (same string as forward pass)
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task)),  # indicators (1=positive, 0=negative)
                    (pair[0][3], pair[1][3]),  # typicality scores
                    (pair[0][4], pair[1][4]),  # is_labeled flags
                    (format_with_inst(pair[0][5].prompt), format_with_inst(pair[1][5].prompt)),  # FIX1: discriminator prompts (chat-templated)
                )
                for pair in pairs_
            ]
        else:
            pairs = [
                (
                    (pair[0][0].prompt, pair[1][0].prompt),  # prompts
                    (pair[0][0].completion, pair[1][0].completion),  # completion for ranking (generator completions)
                    (get_correct_answer(pair[0][2], task), get_correct_answer(pair[1][2], task)),  # validator correct answers
                    (pair[0][0].completion, pair[1][0].completion),  # generator completions (same string as forward pass)
                    (get_indicator(pair[0][2], task), get_indicator(pair[1][2], task)),  # indicators (1=positive, 0=negative)
                    (pair[0][3], pair[1][3]),  # typicality scores
                    (pair[0][4], pair[1][4]),  # is_labeled flags
                    (pair[0][5].prompt, pair[1][5].prompt),  # FIX1: discriminator prompts
                )
                for pair in pairs_
            ]
    elif train_g_or_d == 'both':
        raise NotImplementedError("both mode is not supported in fix1")
    else:
        raise ValueError("TODO!")

    # Diagnostic prints. Guarded against the (extremely unlikely) edge case of
    # len(pairs) < 2 surviving the delta filter; in that case we still want to
    # see the count instead of crashing on pairs[1].
    if len(pairs) == 0:
        # In fix1 g-mode the upstream guard at line ~1864 raises before this
        # point, so this branch is unreachable in practice. Defensive print
        # protects the d/both code paths and any future g-mode refactors.
        raise ValueError("WARNING: pairs is empty after delta filter!")
    else:
        print(pairs[0])
        print("\n\n")
        if len(pairs) >= 2:
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
            self._debug_printed = 0

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            if train_g_or_d == 'both':
                raise NotImplementedError("Not implemented in fix1")
            else:
                # FIX1 (val-NLL position fix, 2026-05-22): 8-element pair structure.
                # 8th element (disc_prompt_i, disc_prompt_j) is the discriminator
                # prompt for each item, used to build a 2nd input sequence
                # `disc_prompt + " Yes"` so val-NLL reads log-odds at the answer
                # slot instead of inside the generator statement. See concern #1
                # in docs/comb_loss_g_mode_concerns.md.
                # Pair structure: (gen_prompts, ranking_completions, validator_correct,
                #                  gen_completions, indicators, typicality, is_labeled,
                #                  disc_prompts)
                (prompt_i, prompt_j), (completion_i, completion_j), (correct_i, correct_j), (gen_completion_i, gen_completion_j), (indicator_i, indicator_j), (typicality_i, typicality_j), (is_labeled_i, is_labeled_j), (disc_prompt_i, disc_prompt_j) = self.pairs[idx]
                
            # Optionally append EOS token text to all completions so both the
            # full-sequence encoding and the separate completion encoding include it.
            if args.include_eos and self.tokenizer.eos_token is not None:
                raise NotImplementedError("Not implemented in fix1")
                _eos = self.tokenizer.eos_token
                if train_g_or_d == 'both':
                    raise NotImplementedError("Not implemented in fix1")
                else:
                    completion_i += _eos
                    completion_j += _eos
                    #TODO: (2026-05-22) --include-eos is not safe with non-log-odds val-NLL
                    # until the discriminator-side tail below also includes EOS.
                    correct_i += _eos
                    correct_j += _eos
                    gen_completion_i += _eos
                    gen_completion_j += _eos

            if train_g_or_d == 'both':
                raise NotImplementedError("Not implemented in fix1")
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

                def _check_tail(name, enc, token_tensor):
                    tail = token_tensor.squeeze(0)
                    tail_len = tail.size(0)
                    if tail_len == 0:
                        return
                    actual = enc['input_ids'].squeeze(0)[-tail_len:]
                    if not torch.equal(actual, tail):
                        raise ValueError(
                            f"{name} tail mismatch (likely truncation/tokenization mismatch). "
                            "Increase --max-seq-len or inspect prompt/completion formatting."
                        )

                _check_tail("generator i", enc_i, token_i)
                _check_tail("generator j", enc_j, token_j)
                if not torch.equal(token_i, token_gen_i) or not torch.equal(token_j, token_gen_j):
                    raise ValueError(
                        "Generator NLL tokens differ from preference completion tokens; "
                        "gen-NLL would read the wrong positions."
                    )

                if nll_validator_weight > 0:
                    # Disc-side input for val-NLL. Always append "Yes"; only its
                    # length is used to locate the answer slot for log-odds.
                    disc_yes_tail = space_prefix + "Yes"
                    input_i_disc = disc_prompt_i + disc_yes_tail
                    input_j_disc = disc_prompt_j + disc_yes_tail
                    enc_i_disc = self.tokenizer(
                        input_i_disc,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )
                    enc_j_disc = self.tokenizer(
                        input_j_disc,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )
                    token_yes_disc = self.tokenizer.encode(disc_yes_tail, add_special_tokens=False, return_tensors='pt')
                    disc_tail = token_yes_disc.squeeze(0)
                    tail_len = disc_tail.size(0)
                    if tail_len > 0:
                        tail_i = enc_i_disc['input_ids'].squeeze(0)[-tail_len:]
                        tail_j = enc_j_disc['input_ids'].squeeze(0)[-tail_len:]
                        if not (torch.equal(tail_i, disc_tail) and torch.equal(tail_j, disc_tail)):
                            raise ValueError(
                                "Discriminator input tail mismatch (likely truncation). "
                                "Increase --max-seq-len or reduce prompt length."
                            )
                    if not validator_log_odds:
                        if token_correct_i.size(1) != token_yes_disc.size(1) or token_correct_j.size(1) != token_yes_disc.size(1):
                            raise ValueError(
                                "Non-log-odds val-NLL requires Yes/No targets to have the same token length "
                                "as the discriminator tail."
                            )

                if debug and self._debug_printed < 3:
                    print(f"[DEBUG dataset] idx={idx}")
                    print(f"[DEBUG dataset] prompt_i={prompt_i[:300]!r}")
                    print(f"[DEBUG dataset] completion_i={completion_i!r}")
                    print(f"[DEBUG dataset] correct_i={correct_i!r} indicator_i={indicator_i} is_labeled_i={is_labeled_i}")
                    print(f"[DEBUG dataset] token_i={token_i.squeeze(0).tolist()} token_gen_i={token_gen_i.squeeze(0).tolist()} token_correct_i={token_correct_i.squeeze(0).tolist()}")
                    if nll_validator_weight > 0:
                        print(f"[DEBUG dataset] disc_prompt_i={disc_prompt_i[:300]!r}")
                        print(f"[DEBUG dataset] token_id_disc={token_yes_disc.squeeze(0).tolist()}")
                    self._debug_printed += 1

            if train_g_or_d != 'both':
                # Squeeze to remove the batch dimension (shape: [seq_len])
                # NOTE/TRAP: this AND-gate is the legacy semantics from the parent
                # script (RP-1 in IMPORTANT-RESEARCH-PLAN §7). It treats a mixed
                # pair (1 labeled + 1 unlabeled) as fully unlabeled, which would
                # silently drop NLL signal that should fire on the labeled side.
                # Fix1's loss block does NOT use this; it uses per-item
                # `is_labeled_i_t` / `is_labeled_j_t` masks instead (set just
                # below). The only reason `pair_is_labeled` survives here is
                # to keep the `'is_labeled'` batch key around for diagnostics /
                # backward compat. If you ever wire the loss block back to
                # `pair_is_labeled`, you re-introduce RP-1.
                # FIX1 (2026-05-22): commented out for stronger guarantee. The
                # value was never read post-line-2756 anyway. Restore by
                # uncommenting both this and the matching `'is_labeled'`
                # entry in `item` below + the load in the train loop.
                # pair_is_labeled = 1.0 if (is_labeled_i and is_labeled_j) else 0.0
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
                    # FIX1 (2026-05-22): legacy AND-gate batch key commented out
                    # in tandem with `pair_is_labeled` above. Was unused by the
                    # active loss block but kept reachable. Restore both lines
                    # together if you ever need the diagnostic.
                    # 'is_labeled': torch.tensor(pair_is_labeled, dtype=torch.float),
                    # FIX1: per-item labeled flags (used by per-item NLL in fix)
                    'is_labeled_i': torch.tensor(1.0 if is_labeled_i else 0.0, dtype=torch.float),
                    'is_labeled_j': torch.tensor(1.0 if is_labeled_j else 0.0, dtype=torch.float),
                }
                if nll_validator_weight > 0:
                    item.update({
                        'input_ids_i_disc': enc_i_disc['input_ids'].squeeze(0),
                        'attention_mask_i_disc': enc_i_disc['attention_mask'].squeeze(0),
                        'input_ids_j_disc': enc_j_disc['input_ids'].squeeze(0),
                        'attention_mask_j_disc': enc_j_disc['attention_mask'].squeeze(0),
                        'token_id_disc': token_yes_disc.squeeze(0),
                    })
            else:
                raise NotImplementedError("Not implemented in fix1")
            return item

    #18 fine for zero-shot
    if use_full_completion:
        batch_size = 1 #TODO: allow actual batches
    else:
        raise NotImplementedError("Not implemented in fix1")

    # FIX1 (batch-size patch, 2026-05-21): apply --batch-size override AFTER the
    # per-task auto-pick block above. Default behavior (no override) is unchanged
    # because args.batch_size is None by default.
    # [DISABLED 2026-05-23] --batch-size flag is commented out in argparse
    # (see ~line 1999). Whole block is dead until we revisit batch_size>1.
    #if args.batch_size is not None and args.batch_size > 0:
    #    raise NotImplementedError("batch size has some issues, so not supported yet")
    #    #if args.batch_size != batch_size:
    #    #    print(f"[--batch-size override] auto-picked={batch_size} -> override={args.batch_size}")
    #    #batch_size = args.batch_size

    dataset = PairwiseDataset(pairs, tokenizer, max_length=max_context_length, device=device, use_full_completion=use_full_completion)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # FIX1 (batch-size patch, 2026-05-21): the parent's raise NotImplementedError
    # for batch_size > 1 + semi-supervised has been LIFTED in fix1 g-mode because:
    #   Bug 1 (BCE batch-mean masking): fixed -- val-NLL log-odds branch now uses
    #     reduction='none' so per-element masking works correctly.
    #   Bug 2 (labeled/unlabeled outer mixing): N/A -- fix1 g-mode removed the
    #     pair_is_labeled * labeled_loss + (1-pair_is_labeled) * unlabeled_loss
    #     branch entirely (single-sum loss, no scalar/[B] mixing).
    # All other loss components (preference, gen-NLL, non-log-odds val-NLL) were
    # already per-element correct because they operate on [B] tensors directly.
    # See unit test in scripts/_smoke_fix1_batch.py for verification.
    print("\n\nDone making dataloader\n\n")
    optimizer = AdamW(model.parameters(), lr=lr)

    losses = []
    global_step = 0

    # Track scores at step 0 (before any training)
    if track_scores:
        raise NotImplementedError("tracking scores is not supported in fix1")

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
                raise NotImplementedError("both mode is not supported in fix1")
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
                # NOTE/TRAP: `pair_is_labeled` is the legacy AND-gate
                # (`is_labeled_i AND is_labeled_j`) inherited from the parent
                # script (RP-1 in IMPORTANT-RESEARCH-PLAN §7). Fix1 does NOT
                # use it in any active loss term -- it's pulled here only for
                # diagnostics / backward compat. Wiring this back into a loss
                # multiplier re-introduces RP-1: mixed pairs (1 labeled + 1
                # unlabeled) get treated as fully unlabeled and silently drop
                # NLL signal that should fire on the labeled side. Always use
                # `is_labeled_i_t` / `is_labeled_j_t` (per-item) below instead.
                # FIX1 (2026-05-22): commented out for stronger guarantee --
                # the `'is_labeled'` key is no longer in the batch dict, so
                # this load would KeyError anyway. Restore alongside the two
                # `__getitem__` writes if you ever need the diagnostic.
                # pair_is_labeled = batch["is_labeled"].to(device)  # 1.0 if both items labeled, 0.0 otherwise (kept for diagnostics)
                # FIX1: per-item is_labeled flags. The fix-mode NLL fires per item, not per pair.
                is_labeled_i_t = batch["is_labeled_i"].to(device)
                is_labeled_j_t = batch["is_labeled_j"].to(device)

                label = batch["label"].to(device)

                # Forward pass for prompt i
                outputs_i = model(input_ids=input_ids_i, attention_mask=attention_mask_i)
                # logits_i: [batch_size, seq_len, vocab_size]

                log_probs_i = F.log_softmax(outputs_i.logits, dim=-1)  # [B, seq_len, vocab_size]

                # Forward pass for prompt j
                outputs_j = model(input_ids=input_ids_j, attention_mask=attention_mask_j)
                log_probs_j = F.log_softmax(outputs_j.logits, dim=-1)  # [B, seq_len, vocab_size]

                if nll_validator_weight > 0:
                    input_ids_i_disc = batch["input_ids_i_disc"].to(device)
                    attention_mask_i_disc = batch["attention_mask_i_disc"].to(device)
                    input_ids_j_disc = batch["input_ids_j_disc"].to(device)
                    attention_mask_j_disc = batch["attention_mask_j_disc"].to(device)
                    token_id_disc = batch["token_id_disc"].to(device)

                    # FIX1 (perf, 2026-05-24): per-item gating of the disc forward
                    # pass. The val-NLL term is `is_labeled_i_t * BCE_i +
                    # is_labeled_j_t * BCE_j` (and analogous in the non-log-odds
                    # branch), so for an unlabeled item the disc-side contribution
                    # is multiplied by zero. Skipping the forward when there is
                    # no labeled item in the batch saves one model(...) call per
                    # such item per training step. Math is bit-identical: zero
                    # times anything (including a non-computed value, here
                    # represented as `None`) is zero.
                    #
                    # Cost: one tiny CPU sync per side (`.any().item()`). Worth
                    # it because we're skipping a full forward+backward pass.
                    #
                    # Expected savings (membership-sans-rosch, semi=0.1):
                    #   per-prompt mode: ~92% both_U + ~8% case_A
                    #     -> ~0.16 disc forwards / pair (down from 2)
                    #     -> ~46% reduction of total (4) forwards per pair
                    #   global mode (default weights): ~67% both_U + ~33% case_A
                    #     -> ~0.66 disc forwards / pair
                    #     -> ~33% reduction of total forwards per pair
                    # For persona-v1 (single prompt, all labeled) and any
                    # task/regime where every item is labeled, the gate is
                    # always taken and behavior is unchanged from before.
                    need_i_disc = bool(is_labeled_i_t.any().item())
                    need_j_disc = bool(is_labeled_j_t.any().item())
                    if need_i_disc:
                        outputs_i_disc = model(input_ids=input_ids_i_disc, attention_mask=attention_mask_i_disc)
                        log_probs_i_disc = F.log_softmax(outputs_i_disc.logits, dim=-1)
                    else:
                        log_probs_i_disc = None
                    if need_j_disc:
                        outputs_j_disc = model(input_ids=input_ids_j_disc, attention_mask=attention_mask_j_disc)
                        log_probs_j_disc = F.log_softmax(outputs_j_disc.logits, dim=-1)
                    else:
                        log_probs_j_disc = None
                else:
                    log_probs_i_disc = None
                    log_probs_j_disc = None
                
                # Helper function to compute log-odds for yes vs no
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
                
                # Compute scores - use log-odds for discriminator mode if flag is set
                if train_g_or_d == 'd' and validator_log_odds:
                    raise NotImplementedError("validator_log_odds is not supported in fix1")
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

                # Use frozen reference model
                if WITH_REF:
                    raise NotImplementedError("with_ref is not supported in fix1")
                else:
                    diff_ref = 0

                # Pairwise logistic loss: - log( sigmoid( (score_j) - (score_i) ) )
                # FIX1: every pair in pair_inds is consistency-filtered (no pair has
                # a labeled item on its wrong natural side), so preference loss can
                # fire on EVERY pair (labeled or not). No labeled/unlabeled gating
                # needed at the pair level.
                diff = score_j - score_i - diff_ref
                preference_loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()

                # FIX1: Validator NLL is PER ITEM, not per pair. The legacy
                # pair_is_labeled outer gate (which fired only when BOTH items
                # were labeled, i.e. only on case_A pairs) is replaced with
                # per-item is_labeled_i_t / is_labeled_j_t masks. Now val NLL
                # fires on every labeled item regardless of its partner:
                #   case_A    -> on both i (L_neg) and j (L_pos)
                #   mixed_neg -> on i only (L_neg; j is unlabeled)
                #   mixed_pos -> on j only (L_pos; i is unlabeled)
                #   both_U    -> nowhere
                # FIX1 (val-NLL position fix, 2026-05-22): val-NLL now reads
                # log-odds from `log_probs_*_disc` (the 2nd forward pass on
                # `disc_prompt + " Yes"`), not from `log_probs_*` (the generator
                # forward pass). This fixes concern #1 in
                # docs/comb_loss_g_mode_concerns.md: pred_pos = -2 of the disc
                # sequence is the actual answer slot where the model predicts
                # Yes/No, instead of a position inside the generator statement.
                # Gated on nll_validator_weight > 0; when 0, the disc forward
                # pass is skipped above and this branch contributes 0.0.
                if nll_validator_weight == 0:
                    nll_validator_loss = torch.tensor(0.0, device=device)
                elif validator_log_odds:
                    # Use log-odds with binary cross-entropy (aligns training with evaluation).
                    # FIX1 (batch-size patch, 2026-05-21): use reduction='none' so the
                    # per-item is_labeled mask is applied PER-ELEMENT, not after the
                    # batch-mean. With reduction='mean' (the default) and B>1 the BCE
                    # collapses to a scalar before the mask, giving every example the
                    # same value -- the mask becomes meaningless. At B=1 the two are
                    # bit-identical (verified by unit test).
                    # FIX1 (val-NLL position fix, 2026-05-22): logits come from
                    # log_probs_i_disc / log_probs_j_disc (disc-prompt forward),
                    # and token_id_disc (= " Yes" tail) defines the slot length.
                    # FIX1 (perf, 2026-05-24): when log_probs_*_disc is None
                    # (per-item gated above because the item is unlabeled), the
                    # corresponding bce term would have been multiplied by
                    # is_labeled_*_t == 0 anyway. Use a zero placeholder; same
                    # numerical result as computing the bce and then zeroing it.
                    if log_probs_i_disc is not None:
                        logodds_correct_i = compute_logodds_simple(log_probs_i_disc, token_id_disc)
                        bce_i = F.binary_cross_entropy_with_logits(logodds_correct_i, indicator_i, reduction='none')
                    else:
                        bce_i = torch.zeros_like(indicator_i)
                    if log_probs_j_disc is not None:
                        logodds_correct_j = compute_logodds_simple(log_probs_j_disc, token_id_disc)
                        bce_j = F.binary_cross_entropy_with_logits(logodds_correct_j, indicator_j, reduction='none')
                    else:
                        bce_j = torch.zeros_like(indicator_j)
                    # FIX1 (2026-05-22): no /2. Each pair contributes per-item
                    # NLL on whichever side(s) are labeled; the surrounding
                    # .mean() already averages across the batch. The legacy /2
                    # was inherited from a parent whose /2 averaged "two items
                    # per pair", but in fix1 most shapes fire on 0 or 1 sides,
                    # so /2 just halved the term for no good reason. Removing
                    # it changes the effective val-NLL scale ~2x relative to
                    # earlier v7 runs -- intentional, the v7 scale wasn't
                    # principled to begin with.
                    nll_validator_loss = (
                        is_labeled_i_t * bce_i + is_labeled_j_t * bce_j
                    ).mean()
                    # For logging, compute score_correct as log-odds (signed by correct answer)
                    #score_correct_i = logodds_correct_i * (2 * indicator_i - 1)
                    #score_correct_j = logodds_correct_j * (2 * indicator_j - 1)
                else:
                    # FIX1 (val-NLL position fix, 2026-05-22): non-log-odds branch
                    # also reads from log_probs_*_disc. Score is now log P(correct_answer | disc_prompt),
                    # i.e. log P("Yes"|disc_prompt) for L+ items and log P("No"|disc_prompt)
                    # for L- items, evaluated AT THE ANSWER SLOT (not somewhere
                    # inside the generator statement, which is what the parent did).
                    #
                    # KNOWN INCONSISTENCY (2026-05-22): the current single-token
                    # gather only credits ONE specific token (e.g. " Yes" for non-chat
                    # or "Yes" for chat models, depending on space_prefix), whereas
                    # eval_by_claude.py aggregates probability across ALL Yes/No
                    # variants in `yestoks` / `notoks` (= ["Yes", " Yes", "YES", ...]).
                    # So train pushes one variant up while eval credits any. The
                    # affected setting is #1 SFT-lo (the only setting in IRP §2 where
                    # val-NLL fires AND --validator-log-odds is OFF). Variants #3, #4,
                    # #7, #11, #12 use --log-odds and go through the branch above which
                    # already aggregates correctly. Variants #2, #5, #6, #8, #9, #10 use
                    # pref-only so val-NLL doesn't fire at all.
                    #
                    # The aggregated version (commented out) below would match eval
                    # semantics. Left disabled to avoid silently changing the SFT-lo
                    # signal mid-experiment. Re-enable by uncommenting the helper +
                    # the three lines below it AND deleting the current two
                    # sum_completion_logprobs calls.
                    #
                    # def _logsumexp_yesno_at_slot(log_probs, token_id_disc, indicator):
                    #     """For each batch element b: return log Σ P(yes_variants) at
                    #     pred_pos when indicator[b]==1, else log Σ P(no_variants).
                    #     Position is identical to compute_logodds_simple, but here we
                    #     gather the gold-side aggregated probability instead of
                    #     log-odds."""
                    #     batch_size = log_probs.shape[0]
                    #     scores = []
                    #     for b in range(batch_size):
                    #         comp_len = token_id_disc[b].size(0)
                    #         pred_pos = -(comp_len + 1)
                    #         probs_at_pos = torch.exp(log_probs[b, pred_pos, :])
                    #         p_yes = probs_at_pos[yestoks].sum()
                    #         p_no = probs_at_pos[notoks].sum()
                    #         p = p_yes if indicator[b].item() >= 0.5 else p_no
                    #         scores.append(torch.log(p + 1e-12))
                    #     return torch.stack(scores)
                    #
                    # score_correct_i = _logsumexp_yesno_at_slot(log_probs_i_disc, token_id_disc, indicator_i)
                    # score_correct_j = _logsumexp_yesno_at_slot(log_probs_j_disc, token_id_disc, indicator_j)
                    # nll_validator_loss = -(is_labeled_i_t * score_correct_i + is_labeled_j_t * score_correct_j).mean()
                    # FIX1 (perf, 2026-05-24): mirror the log-odds branch --
                    # skip score computation if the disc forward was gated off
                    # (log_probs_*_disc is None for unlabeled items).
                    if log_probs_i_disc is not None:
                        score_correct_i = sum_completion_logprobs(log_probs_i_disc, token_correct_i)
                    else:
                        score_correct_i = torch.zeros_like(indicator_i)
                    if log_probs_j_disc is not None:
                        score_correct_j = sum_completion_logprobs(log_probs_j_disc, token_correct_j)
                    else:
                        score_correct_j = torch.zeros_like(indicator_j)
                    nll_validator_loss = -(is_labeled_i_t * score_correct_i + is_labeled_j_t * score_correct_j).mean()

                # FIX1: Generator NLL is per-item, fires only for labeled positives.
                # No pair_is_labeled outer gate. The per-item weighting is:
                #   weight_i = is_labeled_i * indicator_i
                #   weight_j = is_labeled_j * indicator_j
                # For unlabeled items: is_labeled = 0 -> weight = 0.
                # For labeled negatives: indicator = 0 -> weight = 0.
                # For labeled positives: weight = 1.
                # FIX1 (2026-05-22): no /2. With the 4-shape consistent-pair
                # filter, gen-NLL fires on at most ONE side per pair (j on
                # case_A and mixed_pos; nowhere on mixed_neg / both_U). The
                # legacy /2 made sense when the parent could fire on both sides
                # of a case_B (L+/L+) pair, but fix1 drops case_B at
                # construction time. Removing /2 doubles the effective gen-NLL
                # weight relative to v7 fix1 runs -- intentional.
                score_gen_i = sum_completion_logprobs(log_probs_i, token_gen_i)
                score_gen_j = sum_completion_logprobs(log_probs_j, token_gen_j)
                gen_w_i = is_labeled_i_t * indicator_i
                gen_w_j = is_labeled_j_t * indicator_j
                nll_generator_loss = -(gen_w_i * score_gen_i + gen_w_j * score_gen_j).mean()

                # FIX1: Total loss is the sum of pref + (per-item gen-NLL)
                # + (per-item val-NLL). No more "labeled pair vs unlabeled pair"
                # outer branch -- pairs are pre-filtered for consistency at
                # construction time, so preference fires on every pair, gen-NLL
                # fires per-item on labeled positives, and val-NLL fires
                # per-item on every labeled item (positive or negative).
                loss = (
                    preference_loss_weight * preference_loss
                    + nll_validator_weight * nll_validator_loss
                    + nll_generator_weight * nll_generator_loss
                )

                if debug and global_step == 0:
                    dbg(f"batch input_ids_i shape={tuple(input_ids_i.shape)} input_ids_j shape={tuple(input_ids_j.shape)}")
                    dbg(f"token_id_i={token_id_i[0].tolist()} decoded={tokenizer.decode(token_id_i[0])!r}")
                    dbg(f"token_gen_i={token_gen_i[0].tolist()} decoded={tokenizer.decode(token_gen_i[0])!r}")
                    dbg(f"score_i={score_i.detach().cpu().tolist()} score_j={score_j.detach().cpu().tolist()} diff={diff.detach().cpu().tolist()}")
                    if nll_validator_weight > 0:
                        dbg(f"token_id_disc={token_id_disc[0].tolist()} decoded={tokenizer.decode(token_id_disc[0])!r}")
                        if validator_log_odds:
                            dbg(f"logodds_correct_i={logodds_correct_i.detach().cpu().tolist()} logodds_correct_j={logodds_correct_j.detach().cpu().tolist()}")
                        else:
                            dbg(f"score_correct_i={score_correct_i.detach().cpu().tolist()} score_correct_j={score_correct_j.detach().cpu().tolist()}")
                    dbg(
                        "losses "
                        f"pref={float(preference_loss.item()):.6f} "
                        f"val_nll={float(nll_validator_loss.item()):.6f} "
                        f"gen_nll={float(nll_generator_loss.item()):.6f} "
                        f"total={float(loss.item()):.6f}"
                    )
                
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
                    #if nll_validator_weight > 0:
                    #    log_dict["train/score_correct_i"] = score_correct_i.mean().item()
                    #    log_dict["train/score_correct_j"] = score_correct_j.mean().item()
                    if nll_generator_weight > 0:
                        log_dict["train/score_gen_i"] = score_gen_i.mean().item()
                        log_dict["train/score_gen_j"] = score_gen_j.mean().item()
                        log_dict["train/indicator_i"] = indicator_i.mean().item()
                        log_dict["train/indicator_j"] = indicator_j.mean().item()
                    wandb.log(log_dict)
                
                # Track scores for all datapoints at specified frequency
                if track_scores and global_step % track_scores_freq == 0:
                    raise NotImplementedError("tracking scores is not supported in fix1")
               
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
                raise NotImplementedError("iter mode is not supported in fix1")
                direction_str = '--iter'
            elif train_g_or_d == 'both':
                raise NotImplementedError("both mode is not supported in fix1")
                direction_str = '--both'
            else:
                raise ValueError("not supported")

            split_type_str = "--"+ split_type

            alpha_str = "--alpha" + str(alpha) if isinstance(alpha, (int, float)) else "--alpha-" + str(alpha)
            if args.neg_typicality:
                typcorr_str = "--tc-neg"
            elif args.self_typicality:
                typcorr_str = "--tc-self"
            elif args.typicality_correction:
                typcorr_str = "--tc-online"
            else:
                typcorr_str = ""
            lenorm_str = "--lenorm" if args.length_normalize else ""
            single_token_str = "--single-token-data" if args.single_token_data_only else ""
            full_completion_str = "--full-completion" if use_full_completion else ""
            pref_str = f"--pref{preference_loss_weight}" if preference_loss_weight != 1.0 else ""
            nll_v_str = f"--nllv{nll_validator_weight}" if nll_validator_weight > 0 else ""
            nll_g_str = f"--nllg{nll_generator_weight}" if nll_generator_weight > 0 else ""
            force_same_x_str = "--force-same-x" if args.force_same_x else ""
            ppd_str = "--ppd" if args.per_prompt_delta else ""
            cft_str = "--cft" if args.consistency_ft else ""
            valboost_str = "--valboost" if args.boost_initial_val else ""
            vallogodds_str = "--vallogodds" if validator_log_odds else ""
            if args.semi_supervised is not None:
                semi_str = f"--semi{args.semi_supervised}"
            elif args.labeled_only is not None:
                semi_str = f"--labelonly{args.labeled_only}"
            else:
                semi_str = ""
            eos_str = "--eos" if args.include_eos else ""
            # FIX1: append --fix1 suffix so trained checkpoints / score CSVs
            # from the fixed code path are unambiguous.
            fix_str = "--fix1"
            save_directory = args.models_dir + "/v7-" + model_name.replace('/','--')  + f"-delta{delta:.2f}" + "-epoch"+str(epoch) + "--" + task + with_ref_str + all_str + direction_str + split_type_str + alpha_str + typcorr_str + lenorm_str + single_token_str + full_completion_str + eos_str + pref_str + nll_v_str + nll_g_str + force_same_x_str + ppd_str + cft_str + valboost_str + vallogodds_str + semi_str + fix_str
            print("Saving to ", save_directory)
            
            if use_lora:
                # For LoRA: Save adapters first, then optionally merge and save full model.
                print("Saving LoRA adapters...")
                model.save_pretrained(save_directory)

                if args.gemma4_lora:
                    # Skip merge: Gemma 4 31B merge_and_unload() is memory-explosive and
                    # eval_by_claude is configured to load the PEFT adapter directly.
                    print("[--gemma4-lora] Skipping LoRA merge. Adapter saved at:", save_directory)
                    merge_dir = None
                else:
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
                merge_dir = None

            # Upload checkpoint to HuggingFace Hub in a background subprocess
            if not args.no_upload_hf:
                # Adapter-only upload when merge_dir is None (gemma4-lora) or no LoRA at all.
                upload_path = merge_dir if (use_lora and merge_dir is not None) else save_directory
                upload_script = str(Path(__file__).parent.parent / 'src' / 'upload_checkpoint.py')
                cmd = [
                    sys.executable, upload_script,
                    '--local-path', upload_path,
                    '--hf-org', args.hf_org,
                ]
                if args.experiment_notes_dir:
                    cmd += ['--experiment-notes-dir', args.experiment_notes_dir]
                subprocess.Popen(cmd)
                print(f"HF upload started in background: {upload_path}")

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
    #LEGACY_TASKS = ["hypernym", "hypernym-car", "trivia-qa", "swords", "lambada", "ifeval"]
    LEGACY_TASKS = []
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
    parser.add_argument("--delta-bins", type=int, default=None, metavar="N", help="If set, override --delta with auto-computed delta = (p95-p5)/N of validator scores. Logged to <models-dir>/auto_delta_log.csv.")
    parser.add_argument("--per-prompt-delta", action="store_true", default=False,
                        help="[fix1, requires --force-same-x AND --delta-bins] Compute delta "
                             "per prompt as (p95-p5)/N of THAT prompt's validator-score subset, "
                             "instead of one global delta. Use with --force-same-x to avoid "
                             "the global-delta-too-large problem on multi-prompt tasks "
                             "(within-prompt score spreads are typically smaller, so a globally "
                             "calibrated delta over-filters per-prompt pairs). Per-prompt "
                             "deltas are written to the per-run JSON log under <models-dir>/training_run_logs/.")
    parser.add_argument("--total_samples", type=int, default=5110, help="Total samples")
    parser.add_argument("--save_steps", type=int, default=1, help="Save steps")
    parser.add_argument("--all", default=True, action="store_true", help="Whether to use all examples or just positive ones")
    parser.add_argument("--train_g_or_d", type=str, default='g', choices=["d","g","iter","both"], help="Train generator or discriminator.")
    parser.add_argument("--split_type", type=str, default='random', choices=["random","hyper","both"], help="How to do train/test split. Only applies to hypernymy.")
    parser.add_argument("--alpha", type=str, default='1.0', help="Alpha value or function name. NOTE: this is only used when train_g_or_d is 'both'. Can be a number between 0 and 1, or 'alpha_fun_1'")
    parser.add_argument("--lora", action='store_true', help="Use LoRA for memory-efficient fine-tuning")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory (trades compute for memory)")
    parser.add_argument("--typicality-correction", action='store_true', help="Apply typicality correction: use (Generator - GPT-2 P(completion)) instead of raw Generator score")
    parser.add_argument("--self-typicality", action='store_true', help="Use the scoring model itself for typicality correction instead of GPT-2. Implies --typicality-correction.")
    parser.add_argument("--neg-typicality", action='store_true', help="Use negated prompts for typicality correction (LLR: log P(y|Q) - log P(y|neg_Q)). Implies --typicality-correction.")
    parser.add_argument("--no-full-completion", default=False, action='store_true', help="Use only first token for scoring instead of full completion (full completion is default)")
    parser.add_argument("--debug", action='store_true', help="Enable verbose debug output for tokenization checks")
    parser.add_argument("--single_token_data_only", action="store_true", default=False, help="Only use training data where generator completion is exactly one token")
    parser.add_argument("--preference_loss_weight", type=float, default=1.0, help="Weight for preference (pairwise) loss")
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
    parser.add_argument("--semi-supervised", type=float, default=None, metavar="RATIO", help="Semi-supervised training: RATIO (0,1) of prompts are labeled (full loss), rest are unlabeled (preference-only). Mutually exclusive with --labeled-only.")
    parser.add_argument("--labeled-only", type=float, default=None, metavar="RATIO", help="Train only on labeled subset: RATIO (0,1) of prompts are kept, rest discarded. Mutually exclusive with --semi-supervised.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for labeled/unlabeled prompt split (used by --semi-supervised and --labeled-only)")
    parser.add_argument("--disc-shots", type=str, default=None, choices=["zero", "few"], help="Override discriminator shots (default: 'zero' for instruct models, 'few' for base models)")
    parser.add_argument("--include-eos", action="store_true", default=False, help="Append EOS token to completions during training (scores log P(completion+EOS|prompt))")
    parser.add_argument("--gemma4-lora", action="store_true", default=False,
                        help="[Gemma 4 only] Enable Gemma-4-specific LoRA adaptations. "
                             "REQUIRES --lora; the script errors out fast if --gemma4-lora is set "
                             "without --lora. When both are on: "
                             "(a) regex target_modules to find projections wrapped in Gemma4ClippableLinear "
                             "(inner Linear at *.linear), excluding vision_tower; "
                             "(b) skip merge_and_unload() at save time and upload only the adapter dir. "
                             "OFF by default; setting it changes nothing for Gemma 2 / Llama / Qwen runs. "
                             "Required for LoRA training of google/gemma-4-31b-it.")
    parser.add_argument("--models-dir", type=str, default="../models2", help="Directory to save model checkpoints (default: ../models)")
    parser.add_argument("--no-upload-hf", action="store_true", default=False, help="Disable automatic HuggingFace Hub upload after each checkpoint save")
    parser.add_argument("--hf-org", type=str, default="TAUR-dev", help="HuggingFace org to upload checkpoints to")
    parser.add_argument("--experiment-notes-dir", type=str, default="", help="Path to experiment notes dir for updating HUGGINGFACE_REPOS.md")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="N",
        help="Cap training tokenizer max_length (truncate/pad). Use on long-context tasks to reduce VRAM.",
    )
    # FIX1 g-mode pair-shape stratification knobs. Weights are normalized
    # internally to sum to 1; setting any to 0 excludes that shape entirely.
    parser.add_argument("--shape-weight-case-a", type=float, default=0.20,
                        help="[fix1] Sampling weight for case_A pairs (L_neg lo, L_pos hi).")
    parser.add_argument("--shape-weight-mixed-neg", type=float, default=0.20,
                        help="[fix1] Sampling weight for mixed_neg pairs (L_neg lo, U hi).")
    parser.add_argument("--shape-weight-mixed-pos", type=float, default=0.20,
                        help="[fix1] Sampling weight for mixed_pos pairs (U lo, L_pos hi).")
    parser.add_argument("--shape-weight-both-u", type=float, default=0.40,
                        help="[fix1] Sampling weight for both_U pairs (U lo, U hi).")
    parser.add_argument("--consistency-ft", action="store_true", default=False,
                        help="[fix1, SFT-only] Coarse validator/generator consistency filter "
                             "applied BEFORE training. Computes mean validator score t_v and "
                             "mean generator score t_g (raw log P(completion|gen_prompt), no "
                             "length norm, no typicality), binarizes each item by these "
                             "thresholds (1 if score > threshold else 0), and DROPS items "
                             "whose binarized validator and generator labels disagree. "
                             "Requires the SFT setting: --preference_loss_weight 0, "
                             "--nll_validator_weight > 0, --nll_generator_weight > 0, "
                             "and --force-same-x OFF (enforced by argparse). With "
                             "--semi-supervised, only labeled items are filtered; unlabeled "
                             "items pass through unchanged. Adds --cft to save dir name. "
                             "Stats logged to per-run training_run_logs/<...>.json. "
                             "OFF by default (no behavior change).")
    parser.add_argument("--shape-budget-mode", type=str, default="per-prompt",
                        choices=["per-prompt", "global"],
                        help="[fix1] How to allocate the shape sampling budget. "
                             "'per-prompt' (default, original behavior): per-prompt budget "
                             "proportional to n_p, then within-prompt shape stratification "
                             "with within-prompt backfill. Under fsx + prompt-level labeling "
                             "the case_A/both_U mix is determined by the labeled-prompt "
                             "fraction, NOT by the shape weights -- the weights become a no-op. "
                             "'global' (opt-in): allocate per-shape global budgets first "
                             "(weights renormalized over nonzero-pool shapes; deficit "
                             "redistributed to non-saturated shapes); then per-prompt "
                             "allocation within each shape proportional to that prompt's "
                             "pool of that shape. Restores meaningful shape-weight control "
                             "and uses ALL labeled (case_A) pairs every epoch instead of "
                             "~42 percent. Backward-compatible: existing runs that don't pass this "
                             "flag get 'per-prompt' (bit-for-bit unchanged).")
    # [DISABLED 2026-05-23] --batch-size is commented out for now: the
    # per-task auto-pick (default 1 for full-completion mode) is the only
    # supported value. The batch_size>1 path has known issues (see line ~1470)
    # so we don't expose the flag. Re-enable later when those are fixed.
    #parser.add_argument("--batch-size", type=int, default=None, metavar="N",
    #                    help="[fix1] Override training batch size after the per-task auto-pick. "
    #                         "Default: per-task default (usually 1 for full-completion mode). "
    #                         "Lifted the parent's batch_size>1 + semi-supervised guard for fix1 "
    #                         "g-mode -- bug 2 (loss scalar/[B] mixing) was eliminated by fix1's "
    #                         "single-sum loss block; bug 1 (BCE batch-mean masking) was eliminated "
    #                         "by the reduction='none' patch. Use B=2 or B=4 for ~1.5-2x speedup "
    #                         "on 9b-it / 2b-it persona-v1; verify VRAM headroom first.")
    args = parser.parse_args()

    if args.semi_supervised is not None and args.labeled_only is not None:
        parser.error("--semi-supervised and --labeled-only are mutually exclusive")
    for flag_name, flag_val in [("--semi-supervised", args.semi_supervised), ("--labeled-only", args.labeled_only)]:
        if flag_val is not None and not (0 < flag_val < 1):
            parser.error(f"{flag_name} must be between 0 and 1 (exclusive), got {flag_val}")

    if args.neg_typicality and args.self_typicality:
        parser.error("--neg-typicality and --self-typicality are mutually exclusive")

    if args.gemma4_lora and not args.lora:
        parser.error("--gemma4-lora requires --lora (the gemma4-lora flag is a "
                     "modifier for the LoRA path; without --lora the entire LoRA "
                     "block is skipped and gemma4-lora has no effect)")

    if args.per_prompt_delta:
        if not args.force_same_x:
            parser.error("--per-prompt-delta requires --force-same-x (per-prompt deltas "
                         "only make sense when pairs are constructed within prompts).")
        if args.delta_bins is None:
            parser.error("--per-prompt-delta requires --delta-bins N (the per-prompt delta "
                         "is (local_p95 - local_p5) / N; without --delta-bins there is no N).")
    if args.shape_budget_mode == "global" and not args.force_same_x:
        parser.error("--shape-budget-mode global requires --force-same-x. With fsx off "
                     "there is one virtual prompt group, so 'global' and 'per-prompt' "
                     "produce essentially the same allocation; if you really want to "
                     "run fsx-off, use the default --shape-budget-mode per-prompt.")
    if args.consistency_ft:
        if args.preference_loss_weight != 0:
            parser.error("--consistency-ft requires --preference_loss_weight 0 "
                         "(SFT-only filter; preference term must be off).")
        if args.nll_validator_weight <= 0 or args.nll_generator_weight <= 0:
            parser.error("--consistency-ft requires --nll_validator_weight > 0 AND "
                         "--nll_generator_weight > 0 (SFT setting; both NLL terms on).")
        if args.force_same_x:
            parser.error("--consistency-ft requires --force-same-x OFF "
                         "(consistency filter is defined globally; fsx is "
                         "incompatible by design).")
    if args.self_typicality:
        args.typicality_correction = True
    if args.neg_typicality:
        args.typicality_correction = True
    
    # Convert alpha to float if it's a number
    try:
        alpha_val = float(args.alpha)
        if 0 <= alpha_val <= 1:
            args.alpha = alpha_val
    except ValueError:
        if args.alpha not in ["alpha_fun_1"]:
            raise ValueError("Alpha must be a number between 0 and 1, or one of: alpha_fun_1")
    
    main(args)
