#!/usr/bin/env python3
"""
Generate completions for k-SAT prompts using greedy decoding.

Usage:
    # Generator mode (checks if generated assignment satisfies formula)
    CUDA_VISIBLE_DEVICES=0 python generate_ksat.py --model google/gemma-2-2b --task 2sat --mode generator --shots zero --num_samples 10
    
    # Discriminator mode (checks Yes/No exact match)
    CUDA_VISIBLE_DEVICES=0 python generate_ksat.py --model google/gemma-2-2b --task 2sat --mode discriminator --shots few --num_samples 5
"""

import argparse
import os
import sys
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import tasks  # Triggers task registration
from task_registry import get_task


# ============================================================================
# Yes/No probability mass computation
# ============================================================================

def compute_yes_no_probs(model, tokenizer, prompt):
    """
    Compute probability mass for Yes/No token variants.
    Returns dict with individual probs and sums.
    """
    # Token variants to check
    yes_variants = ["Yes", " Yes", "yes", " yes"]
    no_variants = ["No", " No", "no", " no"]
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the last position (next token prediction)
        next_token_logits = outputs.logits[0, -1, :]
        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)
    
    # Get token IDs and their probabilities
    results = {}
    yes_total = 0.0
    no_total = 0.0
    
    for variant in yes_variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if token_ids:
            # Use the first token if multiple
            token_id = token_ids[0]
            prob = probs[token_id].item()
            results[variant] = prob
            yes_total += prob
    
    for variant in no_variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            prob = probs[token_id].item()
            results[variant] = prob
            no_total += prob
    
    results['YES_TOTAL'] = yes_total
    results['NO_TOTAL'] = no_total
    results['TOTAL'] = yes_total + no_total
    
    return results


def print_yes_no_probs(probs):
    """Print Yes/No probabilities in a nice format."""
    print(f">>> YES/NO PROBABILITY MASS:")
    print(f"    Yes variants:")
    for variant in ["Yes", " Yes", "yes", " yes"]:
        if variant in probs:
            print(f"      {repr(variant):8s}: {probs[variant]*100:5.1f}%")
    print(f"    No variants:")
    for variant in ["No", " No", "no", " no"]:
        if variant in probs:
            print(f"      {repr(variant):8s}: {probs[variant]*100:5.1f}%")
    print(f"    -----------------------")
    print(f"    YES total:    {probs['YES_TOTAL']*100:5.1f}%")
    print(f"    NO total:     {probs['NO_TOTAL']*100:5.1f}%")
    print(f"    TOTAL:        {probs['TOTAL']*100:5.1f}%")


# ============================================================================
# SAT checking logic (from check_sat.py)
# ============================================================================

def parse_assignment(assignment_str):
    """Parse assignment string like "x0=1, x1=0, x2=1, ..." """
    assignment = {}
    for match in re.finditer(r'(x\d+)=([01])', assignment_str):
        var = match.group(1)
        val = match.group(2) == '1'
        assignment[var] = val
    return assignment


def parse_literal(lit_str):
    """Parse a literal like "x0" or "¬x1". Returns (var, is_negated)"""
    lit_str = lit_str.strip()
    if lit_str.startswith('¬') or lit_str.startswith('~') or lit_str.startswith('-'):
        return lit_str[1:], True
    return lit_str, False


def parse_clause(clause_str):
    """Parse a clause like "(x0 ∨ ¬x1 ∨ x2)"."""
    clause_str = clause_str.strip().strip('()')
    literals = re.split(r'\s*[∨|]\s*', clause_str)
    return [parse_literal(lit) for lit in literals if lit.strip()]


def parse_formula(formula_str):
    """Parse a CNF formula like "(x0 ∨ x1) ∧ (¬x1 ∨ x2)"."""
    clause_strings = re.split(r'\s*[∧&]\s*', formula_str)
    return [parse_clause(c) for c in clause_strings if c.strip()]


def evaluate_formula(formula, assignment):
    """Evaluate CNF formula. Returns (is_satisfied, failed_clause_idx)."""
    for i, clause in enumerate(formula):
        # Clause is satisfied if any literal is true
        clause_sat = False
        for var, is_negated in clause:
            if var in assignment:
                val = assignment[var]
                lit_val = (not val) if is_negated else val
                if lit_val:
                    clause_sat = True
                    break
        if not clause_sat:
            return False, i
    return True, -1


def check_generated_assignment(formula_str, generated_text):
    """
    Check if generated text contains a valid satisfying assignment.
    Returns (is_correct, parsed_assignment_dict or None)
    """
    # Try to parse assignment from generated text
    assignment = parse_assignment(generated_text)
    
    if not assignment:
        return False, None, "No valid assignment found in output"
    
    # Parse and evaluate formula
    formula = parse_formula(formula_str)
    satisfied, failed_idx = evaluate_formula(formula, assignment)
    
    if satisfied:
        return True, assignment, "Satisfies formula"
    else:
        return False, assignment, f"Fails at clause {failed_idx}"


# ============================================================================
# Main generation logic
# ============================================================================

def main(args):
    print(f"Loading model: {args.model}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Get task config
    task_config = get_task(args.task)
    if task_config is None:
        raise ValueError(f"Task '{args.task}' not found in registry")
    
    # Load data
    print(f"Loading {args.task} data...")
    L_train, L_test = task_config['load_data'](seed=args.seed)
    data = L_test[:args.num_samples]
    
    print(f"\n{'='*80}")
    print(f"Mode: {args.mode.upper()} | Shots: {args.shots} | Task: {args.task}")
    print(f"{'='*80}\n")
    
    correct_count = 0
    
    for i, item in enumerate(data):
        # Get prompt based on mode
        prompt_obj = task_config['make_prompt'](
            item, 
            style=args.mode,  # 'generator' or 'discriminator'
            shots=args.shots
        )
        prompt = prompt_obj.prompt
        ground_truth = prompt_obj.completion.strip()
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,  # Greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Print PROMPT and COMPLETION separately
        print(f"{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"\n>>> PROMPT:\n")
        print(prompt)
        print(f"\n>>> COMPLETION (generated):\n")
        print(generated_text)
        print(f"\n>>> GROUND TRUTH:\n")
        print(ground_truth)
        print()
        
        # Check correctness based on mode
        if args.mode == 'generator':
            # For generator: check if the generated assignment actually satisfies the formula
            is_correct, parsed_assignment, reason = check_generated_assignment(
                item.formula, generated_text
            )
            print(f">>> VERIFICATION:")
            print(f"    Formula: {item.formula}")
            if parsed_assignment:
                # Show just the relevant variables (non-zero ones for brevity)
                active_vars = {k: v for k, v in parsed_assignment.items() if v}
                print(f"    Parsed assignment: {parsed_assignment}")
                print(f"    Active (=1): {active_vars if active_vars else 'none'}")
            print(f"    Result: {reason}")
            print(f"    Correct: {'✓ YES' if is_correct else '✗ NO'}")
            
        else:  # discriminator
            # For discriminator: exact match on Yes/No
            gen_lower = generated_text.strip().lower()
            gt_lower = ground_truth.strip().lower()
            
            # Check if response starts with yes or no
            if gen_lower.startswith('yes'):
                gen_answer = 'yes'
            elif gen_lower.startswith('no'):
                gen_answer = 'no'
            else:
                gen_answer = gen_lower.split()[0] if gen_lower.split() else ''
            
            gt_answer = 'yes' if 'yes' in gt_lower else 'no'
            is_correct = (gen_answer == gt_answer)
            
            print(f">>> VERIFICATION:")
            print(f"    Generated answer: '{gen_answer}'")
            print(f"    Expected answer: '{gt_answer}'")
            print(f"    Correct: {'✓ YES' if is_correct else '✗ NO'}")
            
            # Compute Yes/No probability mass if requested
            if args.compute_yes_no_mass:
                probs = compute_yes_no_probs(model, tokenizer, prompt)
                print()
                print_yes_no_probs(probs)
        
        if is_correct:
            correct_count += 1
        
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Mode: {args.mode} | Shots: {args.shots}")
    print(f"Accuracy: {correct_count}/{len(data)} = {correct_count/len(data):.1%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate k-SAT completions")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b",
                        help="Model to use for generation")
    parser.add_argument("--task", type=str, default="2sat", choices=["2sat", "3sat"],
                        help="Task (2sat or 3sat)")
    parser.add_argument("--mode", type=str, default="generator", 
                        choices=["generator", "discriminator"],
                        help="Mode: generator (produce assignment) or discriminator (yes/no)")
    parser.add_argument("--shots", type=str, default="zero", choices=["zero", "few"],
                        help="Prompt style (zero or few shot)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data loading")
    parser.add_argument("--compute-yes-no-mass", action="store_true",
                        help="Compute and print Yes/No probability mass (discriminator mode only)")
    
    args = parser.parse_args()
    main(args)
