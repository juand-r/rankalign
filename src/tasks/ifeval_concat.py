"""
Concatenated ifeval task that merges all ifeval prompt datasets.
"""

import os
import math
import random

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
import utils


# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
IFEVAL_PROMPT_DATA_DIR = os.path.join(DATA_DIR, 'fixed-prompts-ifeval')

SEED = 0


# Discover and register all available hyponym datasets
def discover_ifeval_datasets():
    """Find all ifeval datasets in the data directory."""
    prompts = set()
    suffix = '.jsonl'
    prefix = 'gpt_ifeval_results_'
    if os.path.exists(IFEVAL_PROMPT_DATA_DIR):
        for filename in os.listdir(IFEVAL_PROMPT_DATA_DIR):
            # Must match exact pattern: gpt_ifeval_results_{name}.jsonl
            if filename.startswith(prefix) and filename.endswith(suffix):
                prompt_name = filename[len(prefix):-len(suffix)]
                if prompt_name:
                    prompts.add(prompt_name)

    return sorted(prompts)


def is_test_only_prompt(prompt_name):
    """True if prompt_1 through prompt_21 (all data goes to test set)."""
    return (
        prompt_name.startswith('prompt_')
        and prompt_name[7:].isdigit()
        and 1 <= int(prompt_name[7:]) <= 21
    )


def load_ifeval_data_raw(prompt_name):
    """Load raw data for a specific prompt from JSONL."""
    jsonl_path = os.path.join(IFEVAL_PROMPT_DATA_DIR, f"gpt_ifeval_results_{prompt_name}.jsonl")
    if not os.path.exists(jsonl_path):
        return []
    return utils.read_data(jsonl_path)


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """
    Load concatenated train/test data from all ifeval prompt datasets.
    
    Uses fixed seed (SEED) for reproducibility, ignoring the passed seed
    to ensure consistent sampling across runs.
    """
    rng = random.Random(SEED)
    
    prompts = discover_ifeval_datasets()
    
    L_train = []
    L_test = []
    
    for prompt_name in prompts:
        dataset = load_ifeval_data_raw(prompt_name)
        if not dataset:
            print(f"[ifeval-concat] Warning: Skipping {prompt_name} - missing data")
            continue

        if is_test_only_prompt(prompt_name):
            # Prompts 1-21: put all in test set
            L_test.extend(dataset)
        else:
            num_train = math.floor(len(dataset) * 0.5)
            train_items, test_items = utils.split_train_test(dataset, seed=SEED, subsample=False, num_train=num_train)

            if not train_items or not test_items:
                print(f"[ifeval-concat] Warning: Skipping {prompt_name} - empty split")
                continue

            L_train.extend(train_items)
            L_test.extend(test_items)
    
    # Shuffle the combined datasets
    rng.shuffle(L_train)
    rng.shuffle(L_test)
    
    print(f"[ifeval-concat] Loaded {len(L_train)} train, {len(L_test)} test from {len(prompts)} datasets")
    
    return L_train, L_test


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """Create prompt for generator or discriminator."""
    return utils.make_prompt_ifeval(
        item,
        style=style,
        shots=shots,
        neg=neg,
        gen_response=gen_response
    )


def get_completion(item):
    """Extract generator completion text."""
    return " " + item["response"].strip()


def get_label(item):
    """Extract binary label ('yes' or 'no')."""
    return item["correct"].strip().lower()


# Register the task
register_task({
    'name': 'ifeval-concat',
    'load_data': load_data,
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    'supports_split_types': ['random'],
    'description': 'Concatenated ifeval task merging all prompt datasets',
})

