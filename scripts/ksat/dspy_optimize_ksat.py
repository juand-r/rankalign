#!/usr/bin/env python3
"""
Use DSPy to search for better prompts for k-SAT tasks.

This script uses DSPy optimizers to find effective prompts/few-shot examples
for the discriminator (validator) task: given a formula and assignment,
determine if the assignment satisfies the formula (Yes/No).

SETUP - Start an SGLang server first:
    # In a separate terminal:
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 \\
        --model-path google/gemma-2-2b --attention-backend triton --disable-cuda-graph

Usage:
    # Evaluate baseline only:
    python dspy_optimize_ksat.py --model google/gemma-2-2b --task 2sat --num_train 25 --num_test 100
    
    # With BootstrapFewShot optimization (works with base models):
    python dspy_optimize_ksat.py --model google/gemma-2-2b --task 2sat --num_train 25 --num_test 100 --optimize
    
    # With MIPROv2 optimization (requires instruct model like gemma-2-2b-it):
    python dspy_optimize_ksat.py --model google/gemma-2-2b-it --task 2sat --optimize --optimizer mipro

OPTIMIZERS:
    - bootstrap (BootstrapFewShot): Selects few-shot examples that work well.
      Simple, works with base models. Does NOT modify instructions.
    
    - mipro (MIPROv2): Generates new instructions AND selects examples.
      More powerful but requires an instruct model that can follow DSPy's
      structured output format with delimiters like [[ ## field ## ]].
"""

import argparse
import os
import sys
import random
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import dspy
import tasks  # Triggers task registration
from task_registry import get_task


# ============================================================================
# DSPy Signatures
# ============================================================================
# These define the input/output schema for our tasks.
# DSPy will use these to generate prompts and parse outputs.

class SATVerifier(dspy.Signature):
    """Determine if a variable assignment satisfies a Boolean formula in CNF."""
    
    formula: str = dspy.InputField(desc="Boolean formula in conjunctive normal form (CNF)")
    assignment: str = dspy.InputField(desc="Variable assignments like x1=0, x2=1, etc.")
    answer: str = dspy.OutputField(desc="Yes if the assignment satisfies the formula, No otherwise")


class SATGenerator(dspy.Signature):
    """Generate a satisfying assignment for a Boolean formula in CNF."""
    
    formula: str = dspy.InputField(desc="Boolean formula in conjunctive normal form (CNF)")
    assignment: str = dspy.OutputField(desc="A satisfying assignment like x1=0, x2=1")


# ============================================================================
# Metrics
# ============================================================================

def discriminator_metric(example, prediction, trace=None):
    """
    Metric for discriminator task.
    Returns 1 if the predicted answer matches ground truth, 0 otherwise.
    Parses Yes/No from freeform text.
    """
    # Get prediction - handle both structured and raw text
    if hasattr(prediction, 'answer'):
        pred_text = prediction.answer or ""
    else:
        pred_text = str(prediction) if prediction else ""
    
    pred_text = pred_text.strip().lower()
    true_answer = example.answer.strip().lower()
    
    # Parse prediction - check first word
    first_word = pred_text.split()[0] if pred_text.split() else ""
    first_word = ''.join(c for c in first_word if c.isalpha())
    
    pred_is_yes = first_word in ['yes', 'y', 'true']
    if not pred_is_yes and first_word not in ['no', 'n', 'false']:
        # Fall back to substring check
        pred_is_yes = 'yes' in pred_text
    
    true_is_yes = true_answer.startswith('yes') or true_answer == 'y'
    
    return 1.0 if (pred_is_yes == true_is_yes) else 0.0


def generator_metric(example, prediction, trace=None):
    """
    Metric for generator task.
    Checks if the generated assignment actually satisfies the formula.
    """
    def parse_assignment(assignment_str):
        assignment = {}
        for match in re.finditer(r'(x\d+)=([01])', assignment_str):
            var = match.group(1)
            val = match.group(2) == '1'
            assignment[var] = val
        return assignment
    
    def parse_literal(lit_str):
        lit_str = lit_str.strip()
        if lit_str.startswith('¬') or lit_str.startswith('~') or lit_str.startswith('-'):
            return lit_str[1:], True
        return lit_str, False
    
    def parse_clause(clause_str):
        clause_str = clause_str.strip().strip('()')
        literals = re.split(r'\s*[∨|]\s*', clause_str)
        return [parse_literal(lit) for lit in literals if lit.strip()]
    
    def parse_formula(formula_str):
        clause_strings = re.split(r'\s*[∧&]\s*', formula_str)
        return [parse_clause(c) for c in clause_strings if c.strip()]
    
    def evaluate_formula(formula, assignment):
        for clause in formula:
            clause_sat = False
            for var, is_negated in clause:
                if var in assignment:
                    val = assignment[var]
                    lit_val = (not val) if is_negated else val
                    if lit_val:
                        clause_sat = True
                        break
            if not clause_sat:
                return False
        return True
    
    # Parse the generated assignment
    pred_assignment = prediction.assignment if hasattr(prediction, 'assignment') else str(prediction)
    assignment = parse_assignment(pred_assignment or "")
    
    if not assignment:
        return 0.0
    
    # Check if it satisfies the formula
    formula = parse_formula(example.formula)
    if evaluate_formula(formula, assignment):
        return 1.0
    return 0.0


# ============================================================================
# DSPy LM Configuration
# ============================================================================

def configure_lm(args):
    """
    Configure DSPy to use a local model server.
    
    For BASE MODELS (text completion, e.g., gemma-2-2b):
        - Use model_type="text" for /v1/completions endpoint
        - Use ChatAdapter with native function calling disabled
        - Model outputs freeform text, DSPy parses using delimiters
    
    For INSTRUCT/CHAT MODELS (e.g., gemma-2-9b-it, llama-3-8b-instruct):
        - Use model_type="chat" for /v1/chat/completions endpoint
        - Can use default ChatAdapter for structured output
        - Model follows chat template
    """
    
    # Auto-detect instruct models by checking for -it or -instruct suffix
    is_instruct = '-it' in args.model.lower() or '-instruct' in args.model.lower()
    
    if is_instruct:
        # -------------------------------------------------------------------------
        # INSTRUCT/CHAT MODEL (e.g., gemma-2-9b-it)
        # -------------------------------------------------------------------------
        print(f"  Detected INSTRUCT model: using model_type='chat'")
        lm = dspy.LM(
            model=f"openai/{args.model}",
            api_base=args.api_base,
            api_key="EMPTY",
            model_type="chat",  # Use chat completions
            max_tokens=150,
            temperature=0.0,
        )
        adapter = dspy.ChatAdapter()
        dspy.configure(lm=lm, adapter=adapter)
    else:
        # -------------------------------------------------------------------------
        # BASE MODEL (text completion, e.g., gemma-2-2b)
        # -------------------------------------------------------------------------
        print(f"  Detected BASE model: using model_type='text'")
        lm = dspy.LM(
            model=f"openai/{args.model}",
            api_base=args.api_base,
            api_key="EMPTY",
            model_type="text",  # Use text completion
            max_tokens=150,
            temperature=0.0,
        )
        # Disable native function calling for base models
        adapter = dspy.ChatAdapter(use_native_function_calling=False)
        dspy.configure(lm=lm, adapter=adapter)
    
    return lm


# ============================================================================
# Main
# ============================================================================

def main(args):
    print("=" * 80)
    print("DSPy Prompt Optimization for k-SAT")
    print("=" * 80)
    
    # Configure DSPy LM
    print(f"\nConfiguring DSPy with model: {args.model}")
    print(f"Connecting to server at: {args.api_base}")
    
    lm = configure_lm(args)
    
    # Get task config and load data
    task_config = get_task(args.task)
    if task_config is None:
        raise ValueError(f"Task '{args.task}' not found in registry")
    
    print(f"\nLoading {args.task} data...")
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
    
    # Create DSPy examples and program based on mode
    if args.mode == 'discriminator':
        print("\nMode: DISCRIMINATOR (Yes/No verification)")
        
        # Split train data into Yes and No examples for balanced demos
        yes_train = [item for item in train_data if item.label.lower() == "yes"]
        no_train = [item for item in train_data if item.label.lower() == "no"]
        
        # Balance the training set (take equal numbers of Yes and No)
        min_count = min(len(yes_train), len(no_train))
        balanced_train = yes_train[:min_count] + no_train[:min_count]
        random.shuffle(balanced_train)
        
        print(f"  Balanced training: {min_count} Yes + {min_count} No = {len(balanced_train)} total")
        
        # Create examples for discriminator with balanced data
        trainset = [
            dspy.Example(
                formula=item.formula,
                assignment=item.assignment,
                answer="Yes" if item.label.lower() == "yes" else "No"
            ).with_inputs('formula', 'assignment')
            for item in balanced_train
        ]
        
        testset = [
            dspy.Example(
                formula=item.formula,
                assignment=item.assignment,
                answer="Yes" if item.label.lower() == "yes" else "No"
            ).with_inputs('formula', 'assignment')
            for item in test_data
        ]
        
        # Create program using dspy.Predict with our signature
        # This allows MIPROv2 to optimize the instructions and few-shot demos!
        program = dspy.Predict(SATVerifier)
        metric = discriminator_metric
        
    else:  # generator
        print("\nMode: GENERATOR (produce satisfying assignment)")
        
        # Create examples for generator (only satisfiable formulas)
        trainset = [
            dspy.Example(
                formula=item.formula,
                assignment=item.satisfying_assignment if hasattr(item, 'satisfying_assignment') and item.satisfying_assignment else item.assignment
            ).with_inputs('formula')
            for item in train_data
            if item.label.lower() == "yes"
        ]
        
        testset = [
            dspy.Example(
                formula=item.formula,
                assignment=item.satisfying_assignment if hasattr(item, 'satisfying_assignment') and item.satisfying_assignment else item.assignment
            ).with_inputs('formula')
            for item in test_data
            if item.label.lower() == "yes"
        ]
        
        # Create program
        program = dspy.Predict(SATGenerator)
        metric = generator_metric
        
        print(f"  (Filtered to satisfiable examples: {len(trainset)} train, {len(testset)} test)")
    
    # Show sample example
    print("\nSample training example:")
    print(f"  Formula: {trainset[0].formula}")
    if args.mode == 'discriminator':
        print(f"  Assignment: {trainset[0].assignment}")
        print(f"  Answer: {trainset[0].answer}")
    else:
        print(f"  Expected assignment: {trainset[0].assignment}")
    
    # Evaluate baseline (zero-shot)
    print("\n" + "=" * 80)
    print("Evaluating BASELINE (zero-shot)...")
    print("=" * 80)
    
    baseline_evaluator = dspy.Evaluate(
        devset=testset[:min(50, len(testset))],  # Use subset for speed
        metric=metric,
        num_threads=1,
        display_progress=True,
        display_table=5,
    )
    baseline_result = baseline_evaluator(program)
    baseline_score = float(baseline_result) if hasattr(baseline_result, '__float__') else getattr(baseline_result, 'score', baseline_result)
    print(f"\nBaseline accuracy: {baseline_score:.1f}%")
    
    if not args.optimize:
        print("\nSkipping optimization (use --optimize to run MIPROv2)")
        return
    
    # Run optimization
    print("\n" + "=" * 80)
    print(f"Running {args.optimizer.upper()} Optimization...")
    print("=" * 80)
    
    if args.optimizer == 'bootstrap':
        # =====================================================================
        # BootstrapFewShot: Simple optimizer for base models
        # =====================================================================
        # This optimizer ONLY selects few-shot examples that produce correct outputs.
        # It does NOT generate new instructions, so it works with base models.
        # Good for: base models, quick optimization, few-shot learning
        from dspy.teleprompt import BootstrapFewShot
        
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=3,  # Max examples to include
            max_labeled_demos=3,        # Max labeled examples to use
            max_rounds=1,               # Number of bootstrap rounds
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=trainset,
        )
        
    elif args.optimizer == 'balanced':
        # =====================================================================
        # Balanced: Manually set balanced Yes/No demos (no optimization)
        # =====================================================================
        # Force exactly N Yes and N No examples as few-shot demos.
        # This prevents the model from being biased toward one answer.
        import copy
        
        # Get Yes and No examples from trainset
        yes_examples = [ex for ex in trainset if ex.answer == "Yes"]
        no_examples = [ex for ex in trainset if ex.answer == "No"]
        
        # Take equal numbers (up to 2 each for 4 total demos)
        num_each = min(2, len(yes_examples), len(no_examples))
        balanced_demos = yes_examples[:num_each] + no_examples[:num_each]
        random.shuffle(balanced_demos)
        
        print(f"  Using {num_each} Yes + {num_each} No = {len(balanced_demos)} balanced demos")
        
        # Create optimized program with manually set demos
        optimized_program = copy.deepcopy(program)
        optimized_program.demos = balanced_demos
        
    elif args.optimizer == 'mipro':
        # =====================================================================
        # MIPROv2: Advanced optimizer that generates new instructions
        # =====================================================================
        # This optimizer generates new instructions AND selects few-shot examples.
        # WARNING: Requires an instruct model to work properly!
        # The model must be able to follow DSPy's delimiter format.
        # Good for: instruct models, thorough optimization, instruction tuning
        from dspy.teleprompt import MIPROv2
        
        optimizer = MIPROv2(
            metric=metric,
            auto='light',  # Use 'light' for faster optimization, 'medium' or 'heavy' for more thorough
            num_threads=1,
            verbose=True,
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Evaluate optimized program
    print("\n" + "=" * 80)
    print("Evaluating OPTIMIZED program...")
    print("=" * 80)
    
    optimized_evaluator = dspy.Evaluate(
        devset=testset,
        metric=metric,
        num_threads=1,
        display_progress=True,
        display_table=10,
    )
    optimized_result = optimized_evaluator(optimized_program)
    optimized_score = float(optimized_result) if hasattr(optimized_result, '__float__') else getattr(optimized_result, 'score', optimized_result)
    print(f"\nOptimized accuracy: {optimized_score:.1f}%")
    print(f"Improvement: {optimized_score - baseline_score:+.1f}%")
    
    # Save the optimized program
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'dspy_optimized_{args.task}_{args.mode}.json')
    optimized_program.save(output_path)
    print(f"\nSaved optimized program to: {output_path}")
    
    # =========================================================================
    # Print the optimized prompt and demos
    # =========================================================================
    print("\n" + "=" * 80)
    print("Optimized Prompt/Instructions:")
    print("=" * 80)
    
    # For dspy.Predict, the optimized instructions are in the signature
    if hasattr(optimized_program, 'signature'):
        sig = optimized_program.signature
        print(f"\nSignature docstring (instructions):")
        print(f"  {sig.__doc__}")
        
        print(f"\nInput fields:")
        for name, field in sig.input_fields.items():
            desc = getattr(field, 'json_schema_extra', {}).get('desc', 'N/A') if hasattr(field, 'json_schema_extra') else 'N/A'
            print(f"  - {name}: {desc}")
        
        print(f"\nOutput fields:")
        for name, field in sig.output_fields.items():
            desc = getattr(field, 'json_schema_extra', {}).get('desc', 'N/A') if hasattr(field, 'json_schema_extra') else 'N/A'
            print(f"  - {name}: {desc}")
    
    # Check for extended signature (MIPROv2 may modify this)
    if hasattr(optimized_program, 'extended_signature'):
        print(f"\nExtended signature instructions:")
        print(f"  {optimized_program.extended_signature}")
    
    # Print demos (few-shot examples chosen by optimizer)
    print("\n" + "=" * 80)
    print("Few-shot Examples (Demos) chosen by optimizer:")
    print("=" * 80)
    
    if hasattr(optimized_program, 'demos') and optimized_program.demos:
        for i, demo in enumerate(optimized_program.demos):
            print(f"\n--- Demo {i+1} ---")
            if hasattr(demo, 'formula'):
                print(f"  Formula: {demo.formula}")
            if hasattr(demo, 'assignment'):
                print(f"  Assignment: {demo.assignment}")
            if hasattr(demo, 'answer'):
                print(f"  Answer: {demo.answer}")
    else:
        print("(No few-shot demos found in optimized_program.demos)")
    
    # Also check for demos in nested predictors
    for attr_name in dir(optimized_program):
        attr = getattr(optimized_program, attr_name, None)
        if hasattr(attr, 'demos') and attr.demos and attr_name != 'demos':
            print(f"\nDemos in {attr_name}:")
            for i, demo in enumerate(attr.demos):
                print(f"\n--- Demo {i+1} ---")
                print(f"  {demo}")
    
    # Try DSPy's inspect_history
    print("\n" + "=" * 80)
    print("Last 5 LM Calls (dspy.inspect_history):")
    print("=" * 80)
    try:
        dspy.inspect_history(n=5)
    except Exception as e:
        print(f"(Not available: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSPy prompt optimization for k-SAT")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b",
                        help="Model to use (must match model served by server)")
    parser.add_argument("--api-base", type=str, default="http://localhost:7501/v1",
                        dest="api_base",
                        help="Server API base URL (default: http://localhost:7501/v1)")
    parser.add_argument("--task", type=str, default="2sat", choices=["2sat", "3sat"],
                        help="Task (2sat or 3sat)")
    parser.add_argument("--mode", type=str, default="discriminator",
                        choices=["discriminator", "generator"],
                        help="Mode: discriminator (yes/no) or generator (produce assignment)")
    parser.add_argument("--num_train", type=int, default=25,
                        help="Number of training examples for optimization")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Number of test examples for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--optimize", action="store_true",
                        help="Run optimization (otherwise just evaluate baseline)")
    parser.add_argument("--optimizer", type=str, default="balanced",
                        choices=["balanced", "bootstrap", "mipro"],
                        help="Optimizer: 'balanced' (force equal Yes/No demos, default), "
                             "'bootstrap' (BootstrapFewShot), 'mipro' (MIPROv2, requires instruct model)")
    
    args = parser.parse_args()
    main(args)
