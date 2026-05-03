#!/usr/bin/env python3
"""
Quick exploration: generate 3-SAT formulas at various difficulty levels,
count satisfying assignments, and produce a small test CSV for eval.

Goal: find parameters where formulas have MULTIPLE correct assignments
(so we can have several correct + several incorrect per problem) and
the verification task isn't trivially easy or impossibly hard.
"""

import random
import math
import csv
from itertools import product
from collections import defaultdict

try:
    from pysat.solvers import Solver
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False


def generate_random_clause(k, n_vars):
    variables = random.sample(range(1, n_vars + 1), k)
    literals = [v if random.random() > 0.5 else -v for v in variables]
    return literals


def generate_formula(k, n_vars, n_clauses):
    return [generate_random_clause(k, n_vars) for _ in range(n_clauses)]


def clause_to_string(clause):
    literals = []
    for lit in clause:
        var_idx = abs(lit) - 1
        if lit > 0:
            literals.append(f"x{var_idx}")
        else:
            literals.append(f"\u00acx{var_idx}")
    return "(" + " \u2228 ".join(literals) + ")"


def formula_to_string(formula):
    return " \u2227 ".join(clause_to_string(c) for c in formula)


def assignment_to_string(assignment, n_vars):
    return ", ".join(f"x{i}={'1' if assignment[i] else '0'}" for i in range(n_vars))


def evaluate_formula(formula, assignment):
    for clause in formula:
        satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0 and assignment[var_idx]:
                satisfied = True
                break
            if lit < 0 and not assignment[var_idx]:
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def count_all_satisfying(formula, n_vars):
    """Brute force count all satisfying assignments (only feasible for small n)."""
    count = 0
    sat_assignments = []
    for bits in product([False, True], repeat=n_vars):
        assignment = {i: bits[i] for i in range(n_vars)}
        if evaluate_formula(formula, assignment):
            count += 1
            sat_assignments.append(assignment)
    return count, sat_assignments


def generate_near_miss(sat_assignment, formula, n_vars, n_flips=1, max_attempts=50):
    """Generate a wrong assignment by flipping variables from a correct one."""
    for _ in range(max_attempts):
        new = sat_assignment.copy()
        to_flip = random.sample(range(n_vars), n_flips)
        for v in to_flip:
            new[v] = not new[v]
        if not evaluate_formula(formula, new):
            return new
    return None


def explore_params(k, n_vars, n_clauses, n_formulas=20, seed=42):
    """Generate formulas and report stats."""
    rng = random.Random(seed)
    random.seed(seed)

    sat_count = 0
    total_solutions = []

    print(f"\n{'='*60}")
    print(f"k={k}, n_vars={n_vars}, n_clauses={n_clauses} (alpha={n_clauses/n_vars:.2f})")
    print(f"{'='*60}")

    for i in range(n_formulas):
        formula = generate_formula(k, n_vars, n_clauses)
        n_sat, sat_assignments = count_all_satisfying(formula, n_vars)
        if n_sat > 0:
            sat_count += 1
            total_solutions.append(n_sat)

    total_possible = 2 ** n_vars
    print(f"  Satisfiable: {sat_count}/{n_formulas} ({100*sat_count/n_formulas:.0f}%)")
    if total_solutions:
        avg = sum(total_solutions) / len(total_solutions)
        print(f"  Solutions per SAT formula: min={min(total_solutions)}, "
              f"avg={avg:.1f}, max={max(total_solutions)}")
        print(f"  (out of {total_possible} possible assignments)")
        # How many have >= 5 solutions?
        multi = sum(1 for s in total_solutions if s >= 5)
        print(f"  Formulas with >=5 solutions: {multi}/{len(total_solutions)}")
    return sat_count, total_solutions


def generate_test_problems(k, n_vars, n_clauses, n_problems=5,
                           min_solutions=3, n_correct=5, n_incorrect=5, seed=42):
    """
    Generate problems suitable for eval: each has multiple correct and incorrect assignments.
    """
    random.seed(seed)
    problems = []

    attempts = 0
    while len(problems) < n_problems and attempts < 500:
        attempts += 1
        formula = generate_formula(k, n_vars, n_clauses)
        n_sat, sat_assignments = count_all_satisfying(formula, n_vars)

        if n_sat < min_solutions:
            continue

        # Pick correct assignments
        correct = random.sample(sat_assignments, min(n_correct, len(sat_assignments)))

        # Generate incorrect (near-miss) assignments
        incorrect = []
        for _ in range(n_incorrect * 3):  # try more than needed
            base = random.choice(sat_assignments)
            n_flips = random.choice([1, 1, 1, 2])  # mostly 1-flip for subtlety
            wrong = generate_near_miss(base, formula, n_vars, n_flips=n_flips)
            if wrong is not None:
                # Avoid duplicates
                wrong_str = assignment_to_string(wrong, n_vars)
                if wrong_str not in [assignment_to_string(a, n_vars) for a in incorrect]:
                    incorrect.append(wrong)
            if len(incorrect) >= n_incorrect:
                break

        if len(incorrect) < 3:
            continue  # not enough wrong answers possible

        problems.append({
            'formula': formula,
            'formula_str': formula_to_string(formula),
            'n_solutions': n_sat,
            'correct': correct,
            'incorrect': incorrect,
        })

    return problems


def write_eval_csv(problems, n_vars, output_path):
    """Write problems as CSV in rankalign ksat format."""
    rows = []
    for p in problems:
        for assign in p['correct']:
            rows.append({
                'formula': p['formula_str'],
                'assignment': assignment_to_string(assign, n_vars),
                'label': 'yes',
                'satisfying_assignment': assignment_to_string(p['correct'][0], n_vars),
            })
        for assign in p['incorrect']:
            rows.append({
                'formula': p['formula_str'],
                'assignment': assignment_to_string(assign, n_vars),
                'label': 'no',
                'satisfying_assignment': assignment_to_string(p['correct'][0], n_vars),
            })

    random.shuffle(rows)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['formula', 'assignment', 'label', 'satisfying_assignment'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows ({sum(1 for r in rows if r['label']=='yes')} yes / "
          f"{sum(1 for r in rows if r['label']=='no')} no) to {output_path}")
    return rows


def main():
    # Explore different parameter combinations
    # For n_vars=10, brute force is feasible (2^10 = 1024 assignments)
    # For n_vars=12, also feasible (2^12 = 4096)
    # For n_vars=15, borderline (2^15 = 32768)

    print("Exploring 3-SAT parameter space...")
    print("Looking for: mostly satisfiable, multiple solutions per formula")

    # Try different alpha values for 3-SAT with 10 vars
    for n_clauses in [20, 25, 30, 35]:
        explore_params(k=3, n_vars=10, n_clauses=n_clauses, n_formulas=30)

    # Try 12 vars
    for n_clauses in [30, 36, 40]:
        explore_params(k=3, n_vars=12, n_clauses=n_clauses, n_formulas=30)

    # Now generate actual test problems at a good difficulty
    # Based on exploration, pick the best parameters
    print("\n" + "="*60)
    print("GENERATING TEST PROBLEMS")
    print("="*60)

    # 10 vars, alpha ~2.5-3.0 should give enough solutions
    problems_10 = generate_test_problems(
        k=3, n_vars=10, n_clauses=25,
        n_problems=5, min_solutions=5,
        n_correct=5, n_incorrect=5, seed=42,
    )
    print(f"\n10-var problems generated: {len(problems_10)}")
    for i, p in enumerate(problems_10):
        print(f"  Problem {i}: {p['n_solutions']} solutions, "
              f"{len(p['correct'])} correct, {len(p['incorrect'])} incorrect assignments")
        print(f"    Formula: {p['formula_str'][:80]}...")

    if problems_10:
        write_eval_csv(problems_10, n_vars=10, output_path='data/3sat_test_explore.csv')

    # Also try 12 vars for harder version
    problems_12 = generate_test_problems(
        k=3, n_vars=12, n_clauses=30,
        n_problems=5, min_solutions=3,
        n_correct=3, n_incorrect=5, seed=42,
    )
    print(f"\n12-var problems generated: {len(problems_12)}")
    for i, p in enumerate(problems_12):
        print(f"  Problem {i}: {p['n_solutions']} solutions, "
              f"{len(p['correct'])} correct, {len(p['incorrect'])} incorrect assignments")

    if problems_12:
        write_eval_csv(problems_12, n_vars=12, output_path='data/3sat_test_explore_12.csv')


if __name__ == "__main__":
    main()
