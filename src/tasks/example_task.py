"""
Example task template.

Copy this file and modify for your new task. Do NOT import this file
in __init__.py - it contains placeholder implementations that will fail.

Instructions:
    1. Copy this file: cp example_task.py my_task.py
    2. Implement all the TODO sections below
    3. Update the register_task() call with your task name
    4. Add `from . import my_task` to __init__.py
    5. Your task is now available via --task my-task
"""

import os
import sys
from collections import namedtuple

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task

# Standard prompt completion namedtuple (must have .prompt and .completion)
PromptCompletion = namedtuple("PromptCompletion", ["prompt", "completion"])


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """
    Load and split data for training/testing.
    
    Args:
        seed: Random seed for shuffling/splitting
        split_type: Type of split ('random', etc.)
        sample_negative: Whether to include negative examples
        **kwargs: Additional task-specific arguments
    
    Returns:
        (L_train, L_test): Lists of data items. Items can be dicts,
                          namedtuples, or any object - just be consistent
                          with what get_completion and get_label expect.
    
    Example implementation:
        import json
        with open('data/my_task.jsonl') as f:
            data = [json.loads(line) for line in f]
        
        random.seed(seed)
        random.shuffle(data)
        
        split_idx = int(len(data) * 0.8)
        return data[:split_idx], data[split_idx:]
    """
    # TODO: Implement data loading
    raise NotImplementedError(
        "Implement load_data() - see docstring for example"
    )


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, **kwargs):
    """
    Create prompt for generator or discriminator.
    
    Args:
        item: Data item from load_data()
        style: 'generator' or 'discriminator'
        shots: 'zero' or 'few' (for few-shot examples)
        gen_response: Optional - use this as the answer in discriminator prompts
                     (for cases where we want to ask about model's own generation)
        neg: Whether to use negation in prompt (for negative examples)
        **kwargs: Additional arguments (e.g., variation)
    
    Returns:
        PromptCompletion namedtuple with:
            .prompt: The input prompt text (will be tokenized)
            .completion: The expected completion (for training/evaluation)
    
    Example implementation:
        if style == 'generator':
            prompt = f"Question: {item['question']}\\n\\nAnswer:"
            completion = " " + item['answer']
        
        elif style == 'discriminator':
            answer = gen_response if gen_response else item['answer']
            prompt = f"Question: {item['question']}\\nProposed answer: {answer}\\nIs this correct?"
            completion = " Yes" if item['is_correct'] else " No"
        
        return PromptCompletion(prompt.strip(), completion)
    """
    # TODO: Implement prompt generation
    if style == 'generator':
        # Generator prompt: ask the model to produce an answer
        # Example: "Complete the sentence: The capital of France is"
        prompt = "YOUR GENERATOR PROMPT HERE"
        completion = " " + "YOUR_COMPLETION"
        
    elif style == 'discriminator':
        # Discriminator prompt: ask Yes/No about a proposed answer
        # Use gen_response if provided, otherwise use ground truth
        answer = gen_response if gen_response else "GROUND_TRUTH_ANSWER"
        prompt = f"YOUR DISCRIMINATOR PROMPT asking about {answer}"
        completion = " Yes"  # or " No" based on item's label
        
    else:
        raise ValueError(f"Unknown style: {style}. Must be 'generator' or 'discriminator'.")
    
    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    """
    Extract generator completion text from data item.
    
    This is the text the generator should produce - typically the answer
    to the question or task.
    
    Args:
        item: Data item from load_data()
    
    Returns:
        str: The completion text. IMPORTANT: Include leading space if the
             completion should be tokenized as a separate word.
             E.g., " Paris" not "Paris"
    
    Example implementations:
        # For dict items:
        return " " + item['answer']
        
        # For namedtuple items:
        return " " + item.answer
        
        # For multi-word completions:
        return " " + item['full_response'].strip()
    """
    # TODO: Implement completion extraction
    # return " " + item['answer']
    raise NotImplementedError(
        "Implement get_completion() - should return ' ' + answer"
    )


def get_label(item):
    """
    Extract binary label from data item.
    
    Args:
        item: Data item from load_data()
    
    Returns:
        str: Must be exactly 'yes' or 'no' (lowercase).
             'yes' = positive/correct example
             'no' = negative/incorrect example
    
    Example implementations:
        # For boolean field:
        return 'yes' if item['is_correct'] else 'no'
        
        # For string field:
        return item['label'].lower()  # assuming 'Yes'/'No' values
        
        # For namedtuple:
        return item.taxonomic.lower()
    """
    # TODO: Implement label extraction
    # return 'yes' if item['is_correct'] else 'no'
    raise NotImplementedError(
        "Implement get_label() - should return 'yes' or 'no'"
    )


# Register this task with the registry
# Uncomment and modify the register_task call when your implementation is ready
#
# register_task({
#     'name': 'example-task',  # Used in --task argument
#     'load_data': load_data,
#     'make_prompt': make_prompt,
#     'get_completion': get_completion,
#     'get_label': get_label,
#     
#     # Optional overrides (uncomment to customize):
#     # 'batch_size': {'with_ref': 1, 'without_ref': 4},
#     # 'supports_negative_sampling': True,
#     # 'supports_split_types': ['random', 'balanced'],
# })

