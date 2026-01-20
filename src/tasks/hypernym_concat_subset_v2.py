"""
Concatenated hypernym task that samples equally from a SUBSET of hypernym-X datasets.
V2 version: Uses grammar-corrected data from fixed-hypernyms/.

This version only includes: bananas, bazookas, cabinets, cars, chairs, crows, diapers, dogs

Variants:
- hypernym-concat-bananas-to-dogs-v2: 89 pos + 89 neg per dataset (1424 train)
- hypernym-concat-bananas-to-dogs-double: 178 pos + 178 neg per dataset (2848 train)
- hypernym-concat-bananas-to-dogs-all: ALL examples per dataset (no train sampling)

All variants share the SAME test set: 68 pos + 68 neg per dataset = 1088 examples
(Achieved by using separate RNGs for train vs test sampling)
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


# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
FIXED_DATA_DIR = os.path.join(DATA_DIR, 'fixed-hypernyms')

# v2 item includes extra fields for grammar-corrected sentences
HypernymItemV2 = namedtuple('HypernymItemV2', [
    'noun1', 'noun2', 'taxonomic',
    'fixed_hypernym_generator', 'discriminator_sentence', 'strategy'
])

# Sampling parameters (base)
TRAIN_POS_PER_DATASET = 89
TRAIN_NEG_PER_DATASET = 89
TEST_POS_PER_DATASET = 68
TEST_NEG_PER_DATASET = 68
SEED = 0

# Fixed subset of hyponyms (bananas to dogs alphabetically)
HYPONYM_SUBSET = [
    'bananas',
    'bazookas',
    'cabinets',
    'cars',
    'chairs',
    'crows',
    'diapers',
    'dogs',
]


def load_hyponym_data_raw(hyponym_name, split='train'):
    """Load raw data for a specific hyponym from fixed CSV (v2 format)."""
    csv_path = os.path.join(FIXED_DATA_DIR, f"hypernym_{hyponym_name}_google-gemma-2-2b_{split}-fixed.csv")
    
    if not os.path.exists(csv_path):
        return []
    
    items = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = HypernymItemV2(
                noun1=row['noun1'],
                noun2=row['fixed_hypernym_generator'],  # Use corrected hypernym
                taxonomic=row['gpt4_ground_truth'].lower(),
                fixed_hypernym_generator=row['fixed_hypernym_generator'],
                discriminator_sentence=row['discriminator_sentence'],
                strategy=row.get('strategy', 'unknown'),
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


def create_load_data_func(train_pos_per_dataset, train_neg_per_dataset, use_all_train=False, task_name=''):
    """
    Factory function to create load_data with specific train sampling parameters.
    
    Test sampling is ALWAYS the same (68 pos + 68 neg per dataset) to ensure
    identical test sets across all variants.
    
    Args:
        train_pos_per_dataset: Number of positive examples to sample per dataset for train
        train_neg_per_dataset: Number of negative examples to sample per dataset for train
        use_all_train: If True, use ALL train examples (no sampling)
        task_name: Name for logging
    """
    def load_data(seed=0, split_type='random', sample_negative=False, v2=True, **kwargs):
        """
        Load concatenated train/test data from the subset of hypernym-X datasets.
        Uses fixed/grammar-corrected data from fixed-hypernyms/.
        
        Uses SEPARATE RNGs for train and test to ensure test sets are identical
        across all variants regardless of train sampling.
        """
        # Separate RNGs for train and test - ensures test set is always the same
        rng_train = random.Random(SEED)
        rng_test = random.Random(SEED)
        
        L_train = []
        L_test = []
        
        for hyponym in HYPONYM_SUBSET:
            # Load raw data from fixed-hypernyms
            train_items = load_hyponym_data_raw(hyponym, 'train')
            test_items = load_hyponym_data_raw(hyponym, 'test')
            
            if not train_items or not test_items:
                print(f"[{task_name}] Warning: Skipping {hyponym} - missing data")
                continue
            
            # Sample train (or use all)
            if use_all_train:
                train_sampled = train_items
            else:
                train_sampled = sample_balanced(train_items, train_pos_per_dataset, train_neg_per_dataset, rng_train)
            
            # Sample test (always the same: 68 pos + 68 neg)
            test_sampled = sample_balanced(test_items, TEST_POS_PER_DATASET, TEST_NEG_PER_DATASET, rng_test)
            
            L_train.extend(train_sampled)
            L_test.extend(test_sampled)
        
        # Shuffle the combined datasets (using separate RNGs)
        rng_train.shuffle(L_train)
        rng_test.shuffle(L_test)
        
        print(f"[{task_name}] Loaded {len(L_train)} train, {len(L_test)} test from {len(HYPONYM_SUBSET)} datasets")
        
        return L_train, L_test
    
    return load_data


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, v2=True, **kwargs):
    """Create prompt for generator or discriminator using v2 grammar-corrected format."""
    return utils.make_prompt_hypernymy_v2(
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


# =============================================================================
# Register task variants
# =============================================================================

# Variant 1: Standard (89 pos + 89 neg per dataset = 1424 train)
register_task({
    'name': 'hypernym-concat-bananas-to-dogs-v2',
    'load_data': create_load_data_func(
        train_pos_per_dataset=TRAIN_POS_PER_DATASET,
        train_neg_per_dataset=TRAIN_NEG_PER_DATASET,
        use_all_train=False,
        task_name='hypernym-concat-bananas-to-dogs-v2'
    ),
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    'supports_split_types': ['random'],
    'description': 'Concatenated hypernym task (v2/fixed) - 89 pos + 89 neg per dataset (1424 train)',
})

# Variant 2: Double (178 pos + 178 neg per dataset = 2848 train)
register_task({
    'name': 'hypernym-concat-bananas-to-dogs-double',
    'load_data': create_load_data_func(
        train_pos_per_dataset=TRAIN_POS_PER_DATASET * 2,
        train_neg_per_dataset=TRAIN_NEG_PER_DATASET * 2,
        use_all_train=False,
        task_name='hypernym-concat-bananas-to-dogs-double'
    ),
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    'supports_split_types': ['random'],
    'description': 'Concatenated hypernym task (v2/fixed) - 178 pos + 178 neg per dataset (2848 train)',
})

# Variant 3: All (use ALL train examples, no sampling)
register_task({
    'name': 'hypernym-concat-bananas-to-dogs-all',
    'load_data': create_load_data_func(
        train_pos_per_dataset=0,  # ignored when use_all_train=True
        train_neg_per_dataset=0,
        use_all_train=True,
        task_name='hypernym-concat-bananas-to-dogs-all'
    ),
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    'supports_split_types': ['random'],
    'description': 'Concatenated hypernym task (v2/fixed) - ALL train examples (no sampling)',
})
