"""
Rosch-1975 category membership tasks (eval-only for now).

NOTE: --neg-typcorr / --neg-typicality IS supported. Even though no
make_negated_prompt is registered here, the legacy dispatch in
eval_by_claude.py:make_negated_gen_prompt handles rosch and membership
tasks by substituting "an example of " -> "an example of something that
is not ".

Each category from Rosch's typicality norms is a separate test task named
'rosch-<category>' (e.g., rosch-furniture, rosch-bird).

Data lives in data/rosch/rosch-<slug>_test.csv with columns:
    category, member, label, generator_sentence, discriminator_sentence,
    rank, similarity_score, distant

Generator prompt uses generator_sentence from the CSV.
Discriminator prompt uses discriminator_sentence from the CSV (code appends " Answer:").
This parallels how hypernym_hyponyms v2 uses discriminator_sentence.
"""

import os
import csv
from collections import namedtuple

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task

PromptCompletion = namedtuple("PromptCompletion", ["prompt", "completion"])

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'rosch'
)

RoschItem = namedtuple('RoschItem', [
    'category', 'member', 'label',
    'generator_sentence', 'discriminator_sentence',
    'rank', 'similarity_score', 'distant',
])


DISC_PREAMBLE = (
    "Do you think bees are dolphins? Answer: No\n\n"
    "Do you think corgis are dogs? Answer: Yes\n\n"
    "Do you think robins are fruit? Answer: No\n\n"
    "Do you think trucks are vehicles? Answer: Yes\n\n"
)


def load_rosch_csv(csv_path):
    items = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(RoschItem(
                category=row['category'],
                member=row['member'],
                label=row['label'].lower(),
                generator_sentence=row['generator_sentence'],
                discriminator_sentence=row['discriminator_sentence'],
                rank=row.get('rank', ''),
                similarity_score=row.get('similarity_score', ''),
                distant=row.get('distant', ''),
            ))
    return items


def create_load_data_func(slug):
    """Factory: creates load_data that returns ([], test_items) since these are eval-only."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        test_path = os.path.join(DATA_DIR, f"rosch-{slug}_test.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Rosch test file not found: {test_path}")
        L_test = load_rosch_csv(test_path)
        return [], L_test
    return load_data


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, **kwargs):
    # TODO: shots parameter is currently ignored; discriminator always includes
    # the preamble and generator is always bare. Wire up shots='zero' to omit
    # the preamble if we want that distinction later.
    if style == 'generator':
        prompt = item.generator_sentence
        completion = " " + item.member

    elif style == 'discriminator':
        prompt = DISC_PREAMBLE + item.discriminator_sentence + " Answer:"
        completion = " " + item.label.capitalize()

    else:
        raise ValueError(f"Unknown style: {style}")

    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    return " " + item.member


def get_label(item):
    return item.label


def discover_rosch_datasets():
    """Find all rosch-*_test.csv files in data/rosch/."""
    slugs = []
    prefix = "rosch-"
    suffix = "_test.csv"
    if os.path.exists(DATA_DIR):
        for filename in sorted(os.listdir(DATA_DIR)):
            if filename.startswith(prefix) and filename.endswith(suffix):
                slug = filename[len(prefix):-len(suffix)]
                if slug:
                    slugs.append(slug)
    return slugs


_registered = []
for slug in discover_rosch_datasets():
    task_name = f"rosch-{slug}"
    try:
        register_task({
            'name': task_name,
            'load_data': create_load_data_func(slug),
            'make_prompt': make_prompt,
            'get_completion': get_completion,
            'get_label': get_label,
            'batch_size': {'with_ref': 1, 'without_ref': 8},
            'supports_split_types': ['random'],
        })
        _registered.append(task_name)
    except Exception as e:
        print(f"[rosch] Warning: Could not register {task_name}: {e}")

if _registered:
    print(f"[rosch] Registered {len(_registered)} tasks: {', '.join(_registered[:5])}{'...' if len(_registered) > 5 else ''}")
