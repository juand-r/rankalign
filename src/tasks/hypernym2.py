"""
Hypernym2 task - exact replica of legacy 'hypernym' task for registry testing.

This task should produce IDENTICAL results to the legacy 'hypernym' task.
Used to verify the task registry system works correctly.
"""

import os
import sys

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task

# Import the existing hypernym functions from utils
# This ensures we use EXACTLY the same logic as the legacy code
import utils


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """
    Load and split hypernym data.
    
    Replicates the legacy behavior exactly:
    - Loads data from load_noun_pair_data()
    - Splits according to split_type: 'random', 'hyper', or 'both'
    """
    L = utils.load_noun_pair_data()
    
    if split_type == 'hyper':
        L_train, L_test = utils.split_train_test_no_overlap(L, seed=seed)
    elif split_type == 'random':
        L_train, L_test = utils.split_train_test(L, seed=seed, subsample=False, num_train=3000)
    elif split_type == 'both':
        L_train, L_test = utils.split_train_test_no_overlap_both(L, seed=2)
    else:
        raise ValueError(f"Wrong value for split_type: {split_type}")
    
    return L_train, L_test


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """
    Create prompt for generator or discriminator.
    
    Directly calls the existing make_prompt_hypernymy function to ensure
    identical behavior.
    """
    return utils.make_prompt_hypernymy(
        item,
        style=style,
        shots=shots,
        neg=neg,
        gen_response=gen_response,
        variation=variation
    )


def get_completion(item):
    """
    Extract generator completion text from data item.
    
    For hypernym, this is the noun2 (the hypernym) with leading space.
    Matches legacy behavior: " " + item.noun2
    """
    return " " + item.noun2


def get_label(item):
    """
    Extract binary label from data item.
    
    For hypernym, the label is stored in item.taxonomic ('yes' or 'no').
    Returns lowercase 'yes' or 'no'.
    """
    return item.taxonomic.strip().lower()


# Register this task with the registry
register_task({
    'name': 'hypernym2',
    'load_data': load_data,
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    
    # Match legacy batch sizes (default is fine here)
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    
    # hypernym supports multiple split types
    'supports_split_types': ['random', 'hyper', 'both'],
})

