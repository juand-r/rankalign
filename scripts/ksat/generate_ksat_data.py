#!/usr/bin/env python3
"""
Generate k-SAT datasets for training and evaluation.

Creates CSV files with k-SAT formulas and variable assignments.
Each row contains a CNF formula and an assignment, with a label
indicating whether the assignment satisfies the formula.

Usage:
    python generate_ksat_data.py --k 2 --n_vars 10 --n_train 3000 --n_test 1000
"""

import argparse
import csv
import random
import math
import os
from collections import namedtuple
from itertools import product

# For SAT solving (optional, for finding satisfying assignments)
try:
    from pysat.solvers import Solver
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False
    print("Warning: pysat not installed. Using brute-force SAT solving (slow for large N).")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate k-SAT dataset")
    parser.add_argument("--k", type=int, default=2, help="Literals per clause (2 for 2-SAT, 3 for 3-SAT)")
    parser.add_argument("--n_vars", type=int, default=10, help="Number of variables (x0, x1, ..., x_{n-1})")
    parser.add_argument("--n_clauses", type=int, default=None, 
                        help="Number of clauses. If not set, auto-computed for ~50%% satisfying assignments")
    parser.add_argument("--n_train", type=int, default=3000, help="Number of training examples")
    parser.add_argument("--n_test", type=int, default=1000, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="../data", help="Output directory for CSVs")
    parser.add_argument("--balance_labels", action="store_true", default=True,
                        help="Balance yes/no labels (half satisfying, half not)")
    return parser.parse_args()


def compute_optimal_clauses(k, target_sat_fraction=0.5):
    """
    Compute number of clauses to achieve target fraction of satisfying assignments.
    
    For m random k-clauses over n variables:
    - Each clause independently satisfied with prob (1 - 1/2^k)
    - Fraction of assignments satisfying all clauses ≈ (1 - 1/2^k)^m
    
    Solving for m: m = log(target) / log(1 - 1/2^k)
    """
    p_clause_sat = 1 - (1 / (2 ** k))
    m = math.log(target_sat_fraction) / math.log(p_clause_sat)
    return max(1, round(m))


def generate_random_clause(k, n_vars):
    """
    Generate a random k-clause over n variables.
    
    Returns a list of literals, where:
    - Positive literal i means x_i
    - Negative literal -i means NOT x_i
    Note: Variables are 1-indexed for SAT solver compatibility (0 is reserved)
    """
    # Sample k distinct variables (1-indexed)
    variables = random.sample(range(1, n_vars + 1), k)
    # Randomly negate each
    literals = [v if random.random() > 0.5 else -v for v in variables]
    return literals


def generate_formula(k, n_vars, n_clauses):
    """Generate a random k-SAT formula as a list of clauses."""
    return [generate_random_clause(k, n_vars) for _ in range(n_clauses)]


def clause_to_string(clause):
    """Convert a clause to human-readable string format."""
    literals = []
    for lit in clause:
        var_idx = abs(lit) - 1  # Convert to 0-indexed for display
        if lit > 0:
            literals.append(f"x{var_idx}")
        else:
            literals.append(f"¬x{var_idx}")
    return "(" + " ∨ ".join(literals) + ")"


def formula_to_string(formula):
    """Convert a formula to human-readable CNF string."""
    return " ∧ ".join(clause_to_string(c) for c in formula)


def assignment_to_string(assignment):
    """
    Convert assignment dict to string format.
    assignment: dict mapping var_idx (0-indexed) to True/False
    """
    parts = []
    for i in sorted(assignment.keys()):
        val = "1" if assignment[i] else "0"
        parts.append(f"x{i}={val}")
    return ", ".join(parts)


def evaluate_clause(clause, assignment):
    """Check if a clause is satisfied by the assignment."""
    for lit in clause:
        var_idx = abs(lit) - 1  # Convert to 0-indexed
        var_value = assignment[var_idx]
        if lit > 0 and var_value:
            return True
        if lit < 0 and not var_value:
            return True
    return False


def evaluate_formula(formula, assignment):
    """Check if all clauses are satisfied."""
    return all(evaluate_clause(c, assignment) for c in formula)


def solve_sat_bruteforce(formula, n_vars):
    """
    Brute-force SAT solver. Returns (is_sat, satisfying_assignment or None).
    """
    for bits in product([False, True], repeat=n_vars):
        assignment = {i: bits[i] for i in range(n_vars)}
        if evaluate_formula(formula, assignment):
            return True, assignment
    return False, None


def solve_sat_pysat(formula, n_vars):
    """
    Use pysat for efficient SAT solving.
    """
    with Solver(name='g3') as solver:
        for clause in formula:
            solver.add_clause(clause)
        
        if solver.solve():
            model = solver.get_model()
            # Convert model to assignment dict
            assignment = {}
            for i in range(1, n_vars + 1):
                assignment[i - 1] = (i in model)
            return True, assignment
        else:
            return False, None


def solve_sat(formula, n_vars):
    """Solve SAT using best available method."""
    if HAS_PYSAT:
        return solve_sat_pysat(formula, n_vars)
    else:
        return solve_sat_bruteforce(formula, n_vars)


def count_satisfying_assignments(formula, n_vars, sample_size=1000):
    """
    Estimate fraction of satisfying assignments by sampling.
    """
    count = 0
    for _ in range(sample_size):
        assignment = {i: random.choice([True, False]) for i in range(n_vars)}
        if evaluate_formula(formula, assignment):
            count += 1
    return count / sample_size


def generate_random_assignment(n_vars):
    """Generate a random variable assignment."""
    return {i: random.choice([True, False]) for i in range(n_vars)}


def flip_assignment(assignment, n_flips=1):
    """Flip n random bits in the assignment to create a non-satisfying one."""
    new_assignment = assignment.copy()
    vars_to_flip = random.sample(list(assignment.keys()), min(n_flips, len(assignment)))
    for var in vars_to_flip:
        new_assignment[var] = not new_assignment[var]
    return new_assignment


def generate_example(k, n_vars, n_clauses, want_satisfying=True, max_attempts=100):
    """
    Generate a single example (formula, assignment, label).
    
    If want_satisfying=True, generates a satisfying assignment.
    If want_satisfying=False, generates a non-satisfying assignment.
    
    Returns: (formula, assignment, label, satisfying_assignment)
    """
    for _ in range(max_attempts):
        formula = generate_formula(k, n_vars, n_clauses)
        is_sat, sat_assignment = solve_sat(formula, n_vars)
        
        if want_satisfying:
            if is_sat:
                return formula, sat_assignment, "yes", sat_assignment
        else:
            # For non-satisfying example:
            if is_sat:
                # Start from satisfying assignment and flip some bits
                non_sat_assignment = flip_assignment(sat_assignment, n_flips=random.randint(1, 3))
                # Verify it's actually non-satisfying
                if not evaluate_formula(formula, non_sat_assignment):
                    return formula, non_sat_assignment, "no", sat_assignment
            else:
                # Formula is unsatisfiable, any assignment is non-satisfying
                random_assignment = generate_random_assignment(n_vars)
                return formula, random_assignment, "no", None
    
    # Fallback: generate any example
    formula = generate_formula(k, n_vars, n_clauses)
    assignment = generate_random_assignment(n_vars)
    label = "yes" if evaluate_formula(formula, assignment) else "no"
    is_sat, sat_assignment = solve_sat(formula, n_vars)
    return formula, assignment, label, sat_assignment


def generate_dataset(k, n_vars, n_clauses, n_examples, balance_labels=True, seed=None, 
                     exclude_pairs=None, max_attempts_per_example=100):
    """
    Generate a dataset of k-SAT examples with NO DUPLICATES.
    
    Args:
        exclude_pairs: Optional set of (formula_str, assignment_str) tuples to exclude
                      (useful for ensuring train/test don't overlap)
    
    Returns list of dicts with keys:
    - formula_str: human-readable CNF formula
    - clauses: list of clause strings
    - assignment_str: human-readable assignment
    - label: "yes" or "no"
    - satisfying_assignment_str: a known satisfying assignment (if exists)
    """
    if seed is not None:
        random.seed(seed)
    
    examples = []
    seen_pairs = set() if exclude_pairs is None else set(exclude_pairs)
    
    def try_add_example(want_satisfying, target_count, label_name):
        """Try to generate unique examples, returns number actually generated."""
        count = 0
        attempts = 0
        max_total_attempts = target_count * max_attempts_per_example
        
        while count < target_count and attempts < max_total_attempts:
            attempts += 1
            formula, assignment, label, sat_assignment = generate_example(
                k, n_vars, n_clauses, want_satisfying=want_satisfying
            )
            
            # Create unique key
            formula_str = formula_to_string(formula)
            assignment_str = assignment_to_string(assignment)
            pair_key = (formula_str, assignment_str)
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                examples.append(make_example_dict(formula, assignment, label, sat_assignment, n_clauses))
                count += 1
                if count % 500 == 0:
                    print(f"  {count}/{target_count} {label_name}")
        
        if count < target_count:
            print(f"  Warning: Only generated {count}/{target_count} unique {label_name} examples (space exhausted)")
        return count
    
    if balance_labels:
        n_positive = n_examples // 2
        n_negative = n_examples - n_positive
        
        print(f"Generating {n_positive} unique positive examples...")
        actual_pos = try_add_example(want_satisfying=True, target_count=n_positive, label_name="positive")
        
        print(f"Generating {n_negative} unique negative examples...")
        actual_neg = try_add_example(want_satisfying=False, target_count=n_negative, label_name="negative")
    else:
        # Random labels - generate unique examples
        count = 0
        attempts = 0
        max_total_attempts = n_examples * max_attempts_per_example
        
        while count < n_examples and attempts < max_total_attempts:
            attempts += 1
            formula = generate_formula(k, n_vars, n_clauses)
            assignment = generate_random_assignment(n_vars)
            
            formula_str = formula_to_string(formula)
            assignment_str = assignment_to_string(assignment)
            pair_key = (formula_str, assignment_str)
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                label = "yes" if evaluate_formula(formula, assignment) else "no"
                is_sat, sat_assignment = solve_sat(formula, n_vars)
                examples.append(make_example_dict(formula, assignment, label, sat_assignment, n_clauses))
                count += 1
                if count % 500 == 0:
                    print(f"  {count}/{n_examples}")
        
        if count < n_examples:
            print(f"  Warning: Only generated {count}/{n_examples} unique examples (space exhausted)")
    
    # Shuffle to mix positive/negative
    random.shuffle(examples)
    return examples, seen_pairs  # Return seen_pairs for train/test exclusion


def make_example_dict(formula, assignment, label, sat_assignment, n_clauses):
    """Create a dictionary for one example."""
    example = {
        'formula': formula_to_string(formula),
        'assignment': assignment_to_string(assignment),
        'label': label,
    }
    
    # Add individual clause columns
    for i, clause in enumerate(formula):
        example[f'clause{i+1}'] = clause_to_string(clause)
    
    # Pad remaining clause columns if formula has fewer clauses
    for i in range(len(formula), n_clauses):
        example[f'clause{i+1}'] = ""
    
    # Add satisfying assignment if known
    if sat_assignment is not None:
        example['satisfying_assignment'] = assignment_to_string(sat_assignment)
    else:
        example['satisfying_assignment'] = ""
    
    return example


def write_csv(examples, filepath, n_clauses):
    """Write examples to CSV file."""
    if not examples:
        return
    
    # Build fieldnames
    fieldnames = ['formula', 'assignment', 'label', 'satisfying_assignment']
    for i in range(n_clauses):
        fieldnames.insert(3 + i, f'clause{i+1}')
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)
    
    print(f"Wrote {len(examples)} examples to {filepath}")


def generate_few_shot_examples(k, n_vars, n_clauses, n_examples=4):
    """
    Generate balanced few-shot examples (2 positive, 2 negative).
    """
    examples = []
    
    # Generate 2 positive
    for _ in range(n_examples // 2):
        formula, assignment, label, sat_assignment = generate_example(
            k, n_vars, n_clauses, want_satisfying=True
        )
        examples.append({
            'formula': formula_to_string(formula),
            'assignment': assignment_to_string(assignment),
            'label': label
        })
    
    # Generate 2 negative
    for _ in range(n_examples - n_examples // 2):
        formula, assignment, label, sat_assignment = generate_example(
            k, n_vars, n_clauses, want_satisfying=False
        )
        examples.append({
            'formula': formula_to_string(formula),
            'assignment': assignment_to_string(assignment),
            'label': label
        })
    
    return examples


def main():
    args = parse_args()
    
    random.seed(args.seed)
    
    # Compute number of clauses if not specified
    if args.n_clauses is None:
        args.n_clauses = compute_optimal_clauses(args.k, target_sat_fraction=0.5)
        print(f"Auto-computed n_clauses={args.n_clauses} for ~50% satisfying assignments with k={args.k}")
    
    print(f"\nGenerating {args.k}-SAT dataset:")
    print(f"  Variables: {args.n_vars} (x0 to x{args.n_vars-1})")
    print(f"  Clauses per formula: {args.n_clauses}")
    print(f"  Training examples: {args.n_train}")
    print(f"  Test examples: {args.n_test}")
    print(f"  Balanced labels: {args.balance_labels}")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate training data (no duplicates within train)
    print("Generating training data...")
    train_examples, train_pairs = generate_dataset(
        args.k, args.n_vars, args.n_clauses, args.n_train,
        balance_labels=args.balance_labels, seed=args.seed
    )
    
    # Generate test data (no duplicates within test, AND no overlap with train)
    print("\nGenerating test data...")
    test_examples, test_pairs = generate_dataset(
        args.k, args.n_vars, args.n_clauses, args.n_test,
        balance_labels=args.balance_labels, seed=args.seed + 1000,
        exclude_pairs=train_pairs  # Exclude all training pairs!
    )
    
    # Write CSVs
    train_path = os.path.join(output_dir, f"{args.k}sat_train.csv")
    test_path = os.path.join(output_dir, f"{args.k}sat_test.csv")
    
    write_csv(train_examples, train_path, args.n_clauses)
    write_csv(test_examples, test_path, args.n_clauses)
    
    # Print statistics
    train_pos = sum(1 for e in train_examples if e['label'] == 'yes')
    test_pos = sum(1 for e in test_examples if e['label'] == 'yes')
    print(f"\nDataset statistics:")
    print(f"  Train: {train_pos}/{len(train_examples)} positive ({100*train_pos/len(train_examples):.1f}%)")
    print(f"  Test: {test_pos}/{len(test_examples)} positive ({100*test_pos/len(test_examples):.1f}%)")
    
    # Generate and print few-shot examples
    print("\n" + "="*60)
    print("FEW-SHOT EXAMPLES (for prompts)")
    print("="*60)
    few_shot = generate_few_shot_examples(args.k, args.n_vars, args.n_clauses, n_examples=4)
    for i, ex in enumerate(few_shot):
        print(f"\nExample {i+1} (label={ex['label']}):")
        print(f"  Formula: {ex['formula']}")
        print(f"  Assignment: {ex['assignment']}")


if __name__ == "__main__":
    main()

