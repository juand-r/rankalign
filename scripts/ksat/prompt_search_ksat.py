#!/usr/bin/env python3
"""
Simple prompt search for k-SAT tasks using local transformers model.

This script evaluates different prompt templates and few-shot configurations
to find the best performing prompts for the discriminator (validator) task.

Usage:
    CUDA_VISIBLE_DEVICES=1 python prompt_search_ksat.py --model google/gemma-2-2b --task 2sat --num_train 25 --num_test 100
"""

import argparse
import os
import sys
import random
import re
from itertools import combinations

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import tasks  # Triggers task registration
from task_registry import get_task


# ============================================================================
# Prompt Templates for Discriminator
# ============================================================================

DISC_TEMPLATES = {
    "simple": """Given the Boolean formula: {formula}
And the variable assignment: {assignment}
Does this assignment satisfy the formula? Answer Yes or No.
Answer:""",

    "explicit": """A Boolean formula in Conjunctive Normal Form (CNF) is satisfied when each clause has at least one true literal.

Formula: {formula}
Assignment: {assignment}

Question: Does the given assignment satisfy the formula?
Answer (Yes/No):""",

    "step_by_step": """To check if an assignment satisfies a CNF formula:
1. Substitute the variable values into each clause
2. A clause is satisfied if at least one literal is true
3. The formula is satisfied if ALL clauses are satisfied

Formula: {formula}
Assignment: {assignment}

Is this formula satisfied? Answer Yes or No:""",

    "compact": """Formula: {formula}
Assignment: {assignment}
Satisfied?""",

    "definition": """A CNF formula φ is satisfied by assignment σ iff every clause in φ contains at least one literal made true by σ.

φ = {formula}
σ = {assignment}

φ(σ) ="""
}

# ============================================================================
# Few-shot example generation
# ============================================================================

def make_few_shot_examples(train_data, num_pos=2, num_neg=2, template_name="simple"):
    """Create few-shot examples from training data."""
    template = DISC_TEMPLATES[template_name]
    
    pos_examples = [item for item in train_data if item.label.lower() == "yes"][:num_pos]
    neg_examples = [item for item in train_data if item.label.lower() == "no"][:num_neg]
    
    examples_str = ""
    for item in pos_examples + neg_examples:
        prompt = template.format(formula=item.formula, assignment=item.assignment)
        answer = " Yes" if item.label.lower() == "yes" else " No"
        examples_str += prompt + answer + "\n\n"
    
    return examples_str


def make_prompt_with_template(item, template_name, few_shot_examples=""):
    """Create a full prompt using a template."""
    template = DISC_TEMPLATES[template_name]
    prompt = template.format(formula=item.formula, assignment=item.assignment)
    if few_shot_examples:
        return few_shot_examples + prompt
    return prompt


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_prompt_config(model, tokenizer, test_data, template_name, few_shot_str=""):
    """Evaluate a prompt configuration on test data."""
    correct = 0
    total = 0
    
    for item in tqdm(test_data, desc=f"Eval {template_name}", leave=False):
        prompt = make_prompt_with_template(item, template_name, few_shot_str)
        
        # Get model's prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
        
        # Parse answer
        if generated_text.startswith('yes'):
            pred = 'yes'
        elif generated_text.startswith('no'):
            pred = 'no'
        else:
            pred = generated_text.split()[0] if generated_text.split() else ''
        
        # Check correctness
        true_answer = item.label.lower().strip()
        if pred == true_answer:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_log_probs(model, tokenizer, test_data, template_name, few_shot_str=""):
    """Evaluate using log-probabilities of Yes vs No."""
    correct = 0
    total = 0
    
    # Get token IDs for Yes and No
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)
    yes_token = yes_tokens[0] if yes_tokens else tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token = no_tokens[0] if no_tokens else tokenizer.encode("No", add_special_tokens=False)[0]
    
    for item in tqdm(test_data, desc=f"LogProb {template_name}", leave=False):
        prompt = make_prompt_with_template(item, template_name, few_shot_str)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token's logits
            
            log_probs = torch.log_softmax(logits, dim=-1)
            log_p_yes = log_probs[yes_token].item()
            log_p_no = log_probs[no_token].item()
        
        # Predict based on which has higher log-prob
        pred = 'yes' if log_p_yes > log_p_no else 'no'
        true_answer = item.label.lower().strip()
        
        if pred == true_answer:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


# ============================================================================
# Main
# ============================================================================

def main(args):
    print("=" * 80)
    print("Prompt Search for k-SAT")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Load data
    task_config = get_task(args.task)
    if task_config is None:
        raise ValueError(f"Task '{args.task}' not found in registry")
    
    print(f"Loading {args.task} data...")
    L_train, L_test = task_config['load_data'](seed=args.seed)
    
    # Shuffle and split
    random.seed(args.seed)
    train_indices = list(range(len(L_train)))
    random.shuffle(train_indices)
    test_indices = list(range(len(L_test)))
    random.shuffle(test_indices)
    
    train_data = [L_train[i] for i in train_indices[:args.num_train]]
    test_data = [L_test[i] for i in test_indices[:args.num_test]]
    
    print(f"Using {len(train_data)} train examples, {len(test_data)} test examples")
    
    # Results storage
    results = []
    
    # Test each template with zero-shot
    print("\n" + "=" * 80)
    print("Evaluating ZERO-SHOT templates...")
    print("=" * 80)
    
    for template_name in DISC_TEMPLATES:
        print(f"\nTesting template: {template_name}")
        
        # Show example prompt
        example_prompt = make_prompt_with_template(train_data[0], template_name)
        print(f"Example prompt:\n{example_prompt}")
        print()
        
        # Evaluate with log-probs
        acc = evaluate_log_probs(model, tokenizer, test_data, template_name)
        print(f"  Accuracy: {acc:.1%}")
        
        results.append({
            'template': template_name,
            'shots': 0,
            'accuracy': acc
        })
    
    # Test templates with few-shot
    print("\n" + "=" * 80)
    print("Evaluating FEW-SHOT templates...")
    print("=" * 80)
    
    shot_configs = [(1, 1), (2, 2), (3, 3)]
    
    for template_name in DISC_TEMPLATES:
        for num_pos, num_neg in shot_configs:
            print(f"\nTesting template: {template_name}, shots: {num_pos}+{num_neg}")
            
            # Create few-shot examples
            few_shot_str = make_few_shot_examples(
                train_data, 
                num_pos=num_pos, 
                num_neg=num_neg,
                template_name=template_name
            )
            
            # Evaluate
            acc = evaluate_log_probs(model, tokenizer, test_data, template_name, few_shot_str)
            print(f"  Accuracy: {acc:.1%}")
            
            results.append({
                'template': template_name,
                'shots': num_pos + num_neg,
                'accuracy': acc
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Template':<20} {'Shots':<8} {'Accuracy':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['template']:<20} {r['shots']:<8} {r['accuracy']:.1%}")
    
    # Best result
    best = results[0]
    print(f"\nBest configuration:")
    print(f"  Template: {best['template']}")
    print(f"  Shots: {best['shots']}")
    print(f"  Accuracy: {best['accuracy']:.1%}")
    
    # Show best prompt
    print("\nBest prompt template:")
    print(DISC_TEMPLATES[best['template']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt search for k-SAT")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b",
                        help="Model to use")
    parser.add_argument("--task", type=str, default="2sat", choices=["2sat", "3sat"],
                        help="Task (2sat or 3sat)")
    parser.add_argument("--num_train", type=int, default=25,
                        help="Number of training examples for few-shot")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Number of test examples for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)

