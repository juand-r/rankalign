"""
Concatenated hypernym task that samples equally from all hypernym-X datasets.

Train: 50 positive + 50 negative from each of 32 datasets = 3200 examples
Test: 17 positive + 17 negative from each of 32 datasets = 1088 examples

This ensures balanced representation across all hyponym categories.
"""

import os
import csv
import random
from collections import namedtuple

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
import utils


# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

# Named tuple matching the expected format
HypernymItem = namedtuple('HypernymItem', ['noun1', 'noun2', 'taxonomic'])

# Sampling parameters
TRAIN_POS_PER_DATASET = 50
TRAIN_NEG_PER_DATASET = 50
TEST_POS_PER_DATASET = 17
TEST_NEG_PER_DATASET = 17
SEED = 0


def discover_hyponym_datasets():
    """Find all hyponym datasets in the data directory."""
    hyponyms = set()
    suffix = '_google-gemma-2-2b_train.csv'
    prefix = 'hypernym_'
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.startswith(prefix) and filename.endswith(suffix):
                hyponym_name = filename[len(prefix):-len(suffix)]
                # Skip 'car' (old file format without model name)
                if hyponym_name and hyponym_name != 'car':
                    hyponyms.add(hyponym_name)
    return sorted(hyponyms)


def load_hyponym_data_raw(hyponym_name, split='train'):
    """Load raw data for a specific hyponym from CSV."""
    csv_path = os.path.join(DATA_DIR, f"hypernym_{hyponym_name}_google-gemma-2-2b_{split}.csv")
    
    if not os.path.exists(csv_path):
        return []
    
    items = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = HypernymItem(
                noun1=row['noun1'],
                noun2=row['predicted_hypernym'],
                taxonomic=row['gpt4_ground_truth'].lower()
            )
            items.append(item)
    
    return items


def sample_balanced(items, n_pos, n_neg, rng):
    """Sample n_pos positive and n_neg negative examples from items."""
    positives = [it for it in items if it.taxonomic == 'yes']
    negatives = [it for it in items if it.taxonomic == 'no']
    
    # Sample with replacement if we don't have enough
    sampled_pos = rng.choices(positives, k=n_pos) if len(positives) < n_pos else rng.sample(positives, n_pos)
    sampled_neg = rng.choices(negatives, k=n_neg) if len(negatives) < n_neg else rng.sample(negatives, n_neg)
    
    return sampled_pos + sampled_neg


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """
    Load concatenated train/test data from all hypernym-X datasets.
    
    Uses fixed seed (SEED) for reproducibility, ignoring the passed seed
    to ensure consistent sampling across runs.
    """
    rng = random.Random(SEED)
    
    hyponyms = discover_hyponym_datasets()
    
    L_train = []
    L_test = []
    
    for hyponym in hyponyms:
        # Load raw data
        train_items = load_hyponym_data_raw(hyponym, 'train')
        test_items = load_hyponym_data_raw(hyponym, 'test')
        
        if not train_items or not test_items:
            print(f"[hypernym-concat] Warning: Skipping {hyponym} - missing data")
            continue
        
        # Sample balanced examples
        train_sampled = sample_balanced(train_items, TRAIN_POS_PER_DATASET, TRAIN_NEG_PER_DATASET, rng)
        test_sampled = sample_balanced(test_items, TEST_POS_PER_DATASET, TEST_NEG_PER_DATASET, rng)
        
        L_train.extend(train_sampled)
        L_test.extend(test_sampled)
    
    # Shuffle the combined datasets
    rng.shuffle(L_train)
    rng.shuffle(L_test)
    
    print(f"[hypernym-concat] Loaded {len(L_train)} train, {len(L_test)} test from {len(hyponyms)} datasets")
    
    return L_train, L_test


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """Create prompt for generator or discriminator."""
    return utils.make_prompt_hypernymy(
        item,
        style=style,
        shots=shots,
        neg=neg,
        gen_response=gen_response,
        variation=variation
    )


def get_completion(item):
    """Extract generator completion text (the hypernym with leading space)."""
    return " " + item.noun2


def get_label(item):
    """Extract binary label ('yes' or 'no')."""
    return item.taxonomic.strip().lower()


# Register the task
register_task({
    'name': 'hypernym-concat',
    'load_data': load_data,
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    'supports_split_types': ['random'],
    'description': 'Concatenated hypernym task sampling equally from all hyponym datasets',
})

