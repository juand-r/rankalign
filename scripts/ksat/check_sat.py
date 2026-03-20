#!/usr/bin/env python3
"""
Check if an assignment satisfies a CNF formula.

Usage:
    python check_sat.py "(x0 ∨ x1) ∧ (¬x1 ∨ x2)" "x0=1, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0"
"""

import re
import sys


def parse_assignment(assignment_str):
    """
    Parse assignment string like "x0=1, x1=0, x2=1, ..."
    Returns dict: {'x0': True, 'x1': False, ...}
    """
    assignment = {}
    # Match patterns like x0=1 or x0=0
    for match in re.finditer(r'(x\d+)=([01])', assignment_str):
        var = match.group(1)
        val = match.group(2) == '1'
        assignment[var] = val
    return assignment


def parse_literal(lit_str):
    """
    Parse a literal like "x0" or "¬x1".
    Returns (variable_name, is_negated)
    """
    lit_str = lit_str.strip()
    if lit_str.startswith('¬'):
        return lit_str[1:], True
    elif lit_str.startswith('~'):
        return lit_str[1:], True
    elif lit_str.startswith('-'):
        return lit_str[1:], True
    else:
        return lit_str, False


def parse_clause(clause_str):
    """
    Parse a clause like "(x0 ∨ ¬x1 ∨ x2)".
    Returns list of (variable_name, is_negated) tuples.
    """
    # Remove parentheses
    clause_str = clause_str.strip()
    if clause_str.startswith('('):
        clause_str = clause_str[1:]
    if clause_str.endswith(')'):
        clause_str = clause_str[:-1]
    
    # Split by OR symbol (∨ or |)
    literals = re.split(r'\s*[∨|]\s*', clause_str)
    
    return [parse_literal(lit) for lit in literals if lit.strip()]


def parse_formula(formula_str):
    """
    Parse a CNF formula like "(x0 ∨ x1) ∧ (¬x1 ∨ x2)".
    Returns list of clauses, where each clause is a list of (var, is_negated).
    """
    # Split by AND symbol (∧ or &)
    clause_strings = re.split(r'\s*[∧&]\s*', formula_str)
    
    return [parse_clause(c) for c in clause_strings if c.strip()]


def evaluate_literal(var, is_negated, assignment):
    """Evaluate a single literal given an assignment."""
    if var not in assignment:
        raise ValueError(f"Variable {var} not in assignment")
    val = assignment[var]
    return (not val) if is_negated else val


def evaluate_clause(clause, assignment):
    """Evaluate a clause (OR of literals). Returns True if any literal is True."""
    for var, is_negated in clause:
        if evaluate_literal(var, is_negated, assignment):
            return True
    return False


def evaluate_formula(formula, assignment):
    """Evaluate a CNF formula (AND of clauses). Returns True if all clauses are True."""
    for i, clause in enumerate(formula):
        if not evaluate_clause(clause, assignment):
            return False, i, clause
    return True, -1, None


def check_sat(formula_str, assignment_str, verbose=True):
    """
    Check if assignment satisfies formula.
    
    Returns: (is_satisfied, details_string)
    """
    assignment = parse_assignment(assignment_str)
    formula = parse_formula(formula_str)
    
    if verbose:
        print(f"Formula: {formula_str}")
        print(f"Assignment: {assignment_str}")
        print(f"Parsed assignment: {assignment}")
        print(f"Parsed formula ({len(formula)} clauses):")
        for i, clause in enumerate(formula):
            clause_str = " ∨ ".join(
                f"{'¬' if neg else ''}{var}" for var, neg in clause
            )
            print(f"  Clause {i}: ({clause_str})")
        print()
    
    satisfied, failed_idx, failed_clause = evaluate_formula(formula, assignment)
    
    if satisfied:
        result = "✓ SATISFIED"
    else:
        clause_str = " ∨ ".join(
            f"{'¬' if neg else ''}{var}={assignment.get(var, '?')}" 
            for var, neg in failed_clause
        )
        result = f"✗ NOT SATISFIED - Clause {failed_idx} fails: ({clause_str})"
    
    if verbose:
        print(result)
    
    return satisfied, result


def main():
    if len(sys.argv) == 3:
        formula_str = sys.argv[1]
        assignment_str = sys.argv[2]
        check_sat(formula_str, assignment_str)
    else:
        # Run on the few-shot examples from ksat.py
        print("=" * 70)
        print("Checking 2-SAT few-shot examples")
        print("=" * 70)
        
        examples = [
            # Example 1 - claimed YES
            {
                'formula': '(x0 ∨ x1) ∧ (¬x1 ∨ x2)',
                'assignment': 'x0=1, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
                'expected': 'yes'
            },
            # Example 2 - claimed NO
            {
                'formula': '(x0 ∨ x1) ∧ (¬x0 ∨ ¬x1)',
                'assignment': 'x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
                'expected': 'no'
            },
            # Example 3 - claimed YES
            {
                'formula': '(¬x0 ∨ x2) ∧ (x1 ∨ ¬x2)',
                'assignment': 'x0=0, x1=1, x2=1, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
                'expected': 'yes'
            },
            # Example 4 - claimed NO
            {
                'formula': '(x0 ∨ x3) ∧ (¬x0 ∨ ¬x3)',
                'assignment': 'x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
                'expected': 'no'
            },
        ]
        
        for i, ex in enumerate(examples):
            print(f"\n--- Example {i+1} (expected: {ex['expected']}) ---")
            satisfied, _ = check_sat(ex['formula'], ex['assignment'])
            actual = 'yes' if satisfied else 'no'
            if actual != ex['expected']:
                print(f"⚠️  MISMATCH: expected {ex['expected']}, got {actual}")
            print()


if __name__ == "__main__":
    main()


