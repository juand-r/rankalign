"""
Category membership training tasks.

Data source (self-contained in this repo):
  data/membership/combined_train_categories_final.json
  data/membership/combined_train_categories_final_negatives.json
  data/membership/excluded_categories.json

Sampling policy for finite-size variants:
  1) Load all categories with separate positive/negative pools.
  2) For large categories (> SMALL_THRESHOLD), sample LARGE_SAMPLE_FRAC from
     each category (separately for pos/neg), deterministically.
  3) For small categories (<= SMALL_THRESHOLD), shuffle category names
     deterministically and add ENTIRE categories until target quota is reached.
     Some small categories will be absent; that is intentional.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import namedtuple

import sys

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task

PromptCompletion = namedtuple("PromptCompletion", ["prompt", "completion"])

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "membership",
)

MembershipItem = namedtuple(
    "MembershipItem", ["category", "member", "label", "similarity", "pool"]
)

CategoryPool = namedtuple("CategoryPool", ["pos", "neg", "all"])

DISC_PREAMBLE = (
    "Do you think bees are dolphins? Answer: No\n\n"
    "Do you think corgis are dogs? Answer: Yes\n\n"
    "Do you think robins are fruit? Answer: No\n\n"
    "Do you think trucks are vehicles? Answer: Yes\n\n"
)

SEED = 42
N_TOTAL_BASE = 2000
SMALL_THRESHOLD = 100
LARGE_SAMPLE_FRAC = 0.20
LARGE_SAMPLE_FRAC_DOUBLE = min(1.0, 2.0 * LARGE_SAMPLE_FRAC)
# Downweight only the two giant categories relative to other large categories.
GIANT_CATEGORY_FRAC_MULTIPLIER = 0.5
GIANT_CATEGORIES = {"food", "animal"}


def _normalize_excluded(s: str) -> str:
    """Strip articles/prefixes for robust exclusion matching."""
    s = s.lower().strip().replace("\x01", "'")
    s = re.sub(r"^(an?|the)\s+", "", s)
    s = re.sub(r"^(article of|type of)\s+", "", s)
    return s


def _article_for(noun: str) -> str:
    """Simple a/an heuristic based on first letter."""
    return "an" if noun and noun[0].lower() in "aeiou" else "a"


def _seed_for(tag: str) -> int:
    """Stable integer seed derived from a tag string."""
    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()
    return SEED + int(digest[:16], 16)


def _deterministic_shuffle(items, tag: str):
    """Return a deterministically shuffled copy of items."""
    out = list(items)
    random.Random(_seed_for(tag)).shuffle(out)
    return out


def _load_category_pools() -> dict[str, CategoryPool]:
    """Load membership data grouped by category with pos/neg split."""
    excl_path = os.path.join(DATA_DIR, "excluded_categories.json")
    pos_path = os.path.join(DATA_DIR, "combined_train_categories_final.json")
    neg_path = os.path.join(DATA_DIR, "combined_train_categories_final_negatives.json")

    with open(excl_path) as f:
        excluded_normalized = {_normalize_excluded(e) for e in json.load(f)}
    with open(pos_path) as f:
        positives = json.load(f)
    with open(neg_path) as f:
        negatives = json.load(f)

    by_cat = {}
    for cat, pos_members in positives.items():
        if _normalize_excluded(cat) in excluded_normalized:
            continue

        pos_items = [
            MembershipItem(
                category=cat,
                member=member,
                label="yes",
                similarity=None,
                pool=None,
            )
            for member in pos_members
        ]

        neg_items = [
            MembershipItem(
                category=cat,
                member=neg_item["example"],
                label="no",
                similarity=neg_item.get("similarity"),
                pool=neg_item.get("pool"),
            )
            for neg_item in negatives.get(cat, [])
        ]

        by_cat[cat] = CategoryPool(
            pos=pos_items,
            neg=neg_items,
            all=pos_items + neg_items,
        )

    return by_cat


def _sample_large_categories(
    by_cat: dict[str, CategoryPool],
    sample_frac: float,
) -> list[MembershipItem]:
    """Sample sample_frac from each large category, deterministically."""
    sampled = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        if len(pool.all) <= SMALL_THRESHOLD:
            continue

        pos_shuf = _deterministic_shuffle(pool.pos, f"large-pos:{cat}")
        neg_shuf = _deterministic_shuffle(pool.neg, f"large-neg:{cat}")

        effective_frac = sample_frac
        if cat in GIANT_CATEGORIES:
            effective_frac = sample_frac * GIANT_CATEGORY_FRAC_MULTIPLIER

        n_pos = max(1, int(round(len(pos_shuf) * effective_frac)))
        n_neg = max(1, int(round(len(neg_shuf) * effective_frac)))
        n_pos = min(n_pos, len(pos_shuf))
        n_neg = min(n_neg, len(neg_shuf))

        sampled.extend(pos_shuf[:n_pos])
        sampled.extend(neg_shuf[:n_neg])

    return sampled


def _add_small_categories_until_quota(
    by_cat: dict[str, CategoryPool],
    items_so_far: list[MembershipItem],
    target_total: int,
) -> list[MembershipItem]:
    """Add ENTIRE small categories until we reach (or slightly exceed) target."""
    selected = list(items_so_far)
    small_cats = [
        cat for cat, pool in by_cat.items()
        if len(pool.all) <= SMALL_THRESHOLD
    ]
    small_cats = _deterministic_shuffle(sorted(small_cats), "small-category-order")

    for cat in small_cats:
        if len(selected) >= target_total:
            break
        selected.extend(by_cat[cat].all)

    return selected


def create_load_data(n_total=None, large_sample_frac=LARGE_SAMPLE_FRAC):
    """Factory for load_data.

    n_total=None -> all items.
    n_total=int  -> sampled variant following the 3-step policy in module docstring.
    """

    def load_data(seed=0, split_type="random", sample_negative=False, **kwargs):
        by_cat = _load_category_pools()

        if n_total is None:
            all_items = [it for pool in by_cat.values() for it in pool.all]
            all_items = _deterministic_shuffle(all_items, "all-items")
            return all_items, []

        large_sample = _sample_large_categories(by_cat, large_sample_frac)
        sampled = _add_small_categories_until_quota(by_cat, large_sample, n_total)
        sampled = _deterministic_shuffle(sampled, f"final:{n_total}")
        return sampled, []

    return load_data


def make_prompt(item, style="generator", shots="zero", gen_response=None, neg=False, **kwargs):
    # TODO: shots parameter is currently ignored; discriminator always includes
    # the preamble and generator is always bare. Wire up shots='zero' to omit
    # the preamble if we want that distinction later.
    cat = item.category
    member = item.member
    article = _article_for(cat)

    if style == "generator":
        prompt = f"Complete the sentence: an example of {article} {cat} is"
        completion = " " + member

    elif style == "discriminator":
        question = f"Do you think {member} is {article} {cat}? Answer:"
        prompt = DISC_PREAMBLE + question
        completion = " Yes" if item.label == "yes" else " No"

    else:
        raise ValueError(f"Unknown style: {style}")

    return PromptCompletion(prompt, completion)


def get_completion(item):
    return " " + item.member


def get_label(item):
    return item.label


# =============================================================================
# Register task variants
# =============================================================================

_COMMON = dict(
    make_prompt=make_prompt,
    get_completion=get_completion,
    get_label=get_label,
    batch_size={"with_ref": 1, "without_ref": 8},
    supports_split_types=["random"],
)

register_task({
    "name": "membership-sans-rosch-v0",
    "load_data": create_load_data(
        n_total=N_TOTAL_BASE,
        large_sample_frac=LARGE_SAMPLE_FRAC,
    ),
    **_COMMON,
})

register_task({
    "name": "membership-sans-rosch-v0-double",
    "load_data": create_load_data(
        n_total=N_TOTAL_BASE * 2,
        large_sample_frac=LARGE_SAMPLE_FRAC_DOUBLE,
    ),
    **_COMMON,
})

register_task({
    "name": "membership-sans-rosch-v0-all",
    "load_data": create_load_data(n_total=None),
    **_COMMON,
})
