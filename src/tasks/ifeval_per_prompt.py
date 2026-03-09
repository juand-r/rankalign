"""
IFEval tasks for specific prompts.
"""

import os
import csv
from collections import namedtuple
import math

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


# Named tuple matching the expected format
HypernymItem = namedtuple('HypernymItem', ['noun1', 'noun2', 'taxonomic'])

# v2 item includes extra fields for grammar-corrected sentences
HypernymItemV2 = namedtuple('HypernymItemV2', [
    'noun1', 'noun2', 'taxonomic',
    'fixed_hypernym_generator', 'discriminator_sentence', 'strategy'
])
    


def create_load_data_func(prompt_name):
    """Factory function to create load_data for a specific prompt."""
    def load_data(seed=0, split_type='random', sample_negative=False, v2=True, **kwargs):
        """Load train/test data for this prompt with a fixed split."""
        dataset = utils.read_data('../data/fixed-prompts-ifeval/gpt_ifeval_results_{}.jsonl'.format(prompt_name))

        if prompt_name and (prompt_name.startswith('prompt_') and prompt_name[7:].isdigit() and 1 <= int(prompt_name[7:]) <= 21):
            num_train = 0
        else:
            num_train = math.floor(len(dataset) * 0.5)
        return utils.split_train_test(dataset, seed=SEED, subsample=False, num_train=num_train)
    return load_data


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, v2=True, **kwargs):
    """
    Create prompt for generator or discriminator.
    """
    return utils.make_prompt_ifeval(
        item,
        style=style,
        shots=shots,
        neg=neg,
        gen_response=gen_response
    )


def get_completion(item):
    """Extract generator completion text (the hypernym with leading space)."""
    return " " + item["response"].strip()


def get_label(item):
    """Extract binary label ('yes' or 'no')."""
    return item["correct"].lower()


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


# Register each prompt as a separate task
_registered_prompts = []
for prompt_name in discover_ifeval_datasets():
    task_name = f'ifeval-{prompt_name}'
    
    # Skip if already registered (e.g., hypernym-cars exists elsewhere)
    try:
        register_task({
            'name': task_name,
            'load_data': create_load_data_func(prompt_name),
            'make_prompt': make_prompt,
            'get_completion': get_completion,
            'get_label': get_label,
            'batch_size': {'with_ref': 1, 'without_ref': 6},
            'supports_split_types': ['random'],  # These datasets have fixed train/test splits
            'description': f'IFEval task for {prompt_name}',
        })
        _registered_prompts.append(task_name)
    except Exception as e:
        print(f"[ifeval_per_prompt] Warning: Could not register {task_name}: {e}")

if _registered_prompts:
    print(f"[ifeval_per_prompt] Registered {len(_registered_prompts)} tasks: {', '.join(_registered_prompts[:5])}{'...' if len(_registered_prompts) > 5 else ''}")