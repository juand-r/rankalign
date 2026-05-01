# Rosch Dataset Creation — Step-by-Step Record

This document records every step taken to create the Rosch-1975 category membership eval tasks, so the process can be replicated for training data.

---

## Overview

- **Task family**: `rosch-<category>` (e.g., `rosch-furniture`, `rosch-bird`)
- **Purpose**: eval-only test sets for category membership
- **Source data**: lives in `/datastor1/jdr/gv-gap/flora/`
- **Output data**: lives in `data/rosch/` within the rankalign repo
- **Task module**: `src/tasks/rosch.py`

---

## Source Files (in flora repo)

1. **`flora/rosch-1975.json`** — Main Rosch 1975 typicality norms
   - Structure: `{"title": "...", "categories": {"Furniture": [{"member": "chair", "rank": 1.5, "specific_score": 1.04, "correct": "yes", "generator_sentence": "...", "discriminator_sentence": "..."}, ...]}}`
   - 10 categories, 565 total items
   - Each item has `correct: "yes"` or `"no"` (original Rosch judgments)
   - We added `generator_sentence` and `discriminator_sentence` fields to every item

2. **`flora/incorrect_items-to-supplement-rosch-75.json`** — Extra negatives
   - Structure: `{"title": "...", "categories": {"Furniture": [{"member": "blanket", "distant": "false", "generator_sentence": "...", "discriminator_sentence": "..."}, ...]}}`
   - Provides additional "no" items per category (especially for Bird which had almost none)
   - Each item has `distant: "true"` or `"false"` indicating whether it's a far negative
   - We added `generator_sentence` and `discriminator_sentence` fields here too

---

## Step 1: Add `generator_sentence` and `discriminator_sentence` to the source JSONs

Script: `scripts/add_sentences_to_rosch_jsons.py`

This one-time script reads both JSONs and adds two fields per item:

- **`generator_sentence`**: The prompt prefix for the generator. Examples:
  - `"Complete the sentence: an example of furniture is a"` (singular countable member)
  - `"Complete the sentence: an example of a sport is"` (activity/mass noun member — no article)
  - `"Complete the sentence: an example of clothing is"` (plural member — no article)
  - `"Complete the sentence: an example of a fruit is an"` (vowel-initial member)

- **`discriminator_sentence`**: The yes/no question (WITHOUT "Answer:" — that gets appended by the task code). Examples:
  - `"Do you think a chair is furniture?"` (singular countable)
  - `"Do you think football is a sport?"` (activity — no article)
  - `"Do you think pants are clothing?"` (plural — uses "are")
  - `"Do you think an orange is a fruit?"` (vowel-initial)

### Grammar logic in the script:

Three-way classification of members:
- **Singular** (default): gets "a/an" article, uses "is". Example: "a chair is furniture"
- **Plural** (hardcoded set): no article, uses "are". Example: "pants are clothing"
- **No-article** (hardcoded set): no article, uses "is". Example: "billiards is a sport", "football is a sport"

Category articles:
- Uncountable categories (`furniture`, `clothing`): no article. "an example of furniture is"
- Countable categories: get "a/an". "an example of a bird is"

The plural/no-article sets are hardcoded in the script (see `PLURAL_MEMBERS` and `NO_ARTICLE_ITEMS`).

After running the script, we manually reviewed and fixed remaining issues (56 items fixed: sports activities, mass noun vegetables like parsley/corn/spinach, materials like wood/sandpaper/cotton).

---

## Step 2: Add Bird negatives to the supplement JSON

The original Rosch data had only 1 negative for Bird ("bat"). We added 52 items to `incorrect_items-to-supplement-rosch-75.json` under the "Bird" category:

- **18 flying things** (close negatives, `distant: "false"`): fruit bat, butterfly, moth, dragonfly, damselfly, flying squirrel, flying fish, sugar glider, flying fox, bee, wasp, hornet, mosquito, firefly, ladybug, cicada, locust, pterodactyl
- **12 bird-related** (close, `distant: "false"`): egg, nest, feather, birdhouse, aviary, birdwatcher, birdseed, birdcage, perch, wingspan, talon, beak
- **16 bird-name idioms** (close, `distant: "false"`): bird of paradise, crow's feet, goosebumps, nest egg, crow bar, swallow tail, dove bar, eagle scout, crane operator, robin hood, jay walking, lark about, swan lake, birdie putt, cardinal sin, albatross around your neck
- **6 far negatives** (`distant: "true"`): airplane, kite, helicopter, frisbee, boomerang, rocket

Important: we checked for overlaps with the Rosch Bird positives. "bat" was already in rosch-1975.json as the original negative, so we used "fruit bat" instead.

---

## Step 3: Generate the test CSVs

Script: `scripts/make_rosch_test_sets.py`

This script reads both JSONs and outputs one CSV per category to `data/rosch/rosch-<slug>_test.csv`.

Slug conversion: `category_to_task_slug()` lowercases, replaces `"'s "` with `"s-"`, replaces spaces with `"-"`.
- "Carpenter's tool" → "carpenters-tool"
- "Bird" → "bird"

CSV columns: `category, member, label, generator_sentence, discriminator_sentence, rank, similarity_score, distant`

For each category:
1. All items from rosch-1975.json go in (both yes and no)
2. All items from the supplement JSON go in (all labeled "no")
3. `generator_sentence` and `discriminator_sentence` are passed through directly from the JSON — no logic in this script

Final counts:
- rosch-furniture: 40 pos + 40 neg = 80
- rosch-fruit: 42 pos + 42 neg = 84
- rosch-vehicle: 41 pos + 41 neg = 82
- rosch-weapon: 42 pos + 42 neg = 84
- rosch-vegetable: 43 pos + 43 neg = 86
- rosch-carpenters-tool: 41 pos + 41 neg = 82
- rosch-bird: 53 pos + 53 neg = 106
- rosch-sport: 46 pos + 46 neg = 92
- rosch-toy: 36 pos + 36 neg = 72
- rosch-clothing: 44 pos + 44 neg = 88

---

## Step 4: Create the task module

File: `src/tasks/rosch.py`

Key design decisions (paralleling `src/tasks/hypernym_hyponyms.py` and `src/utils.py:make_prompt_hypernymy_v2`):

- **Item type**: `RoschItem` namedtuple with fields: `category, member, label, generator_sentence, discriminator_sentence, rank, similarity_score, distant`

- **`load_data()`**: Returns `([], L_test)` — empty train, full test. Eval-only for now.

- **`make_prompt()`**:
  - Generator: `prompt = item.generator_sentence`, `completion = " " + item.member`
  - Discriminator: `prompt = DISC_PREAMBLE + item.discriminator_sentence + " Answer:"`, `completion = " Yes" or " No"`
  - The code appends `" Answer:"` (NOT stored in the JSON/CSV) — matching how hypernymy v2 does it.

- **`DISC_PREAMBLE`**: Same as hypernymy v2:
  ```
  Do you think bees are dolphins? Answer: No
  Do you think corgis are dogs? Answer: Yes
  Do you think robins are fruit? Answer: No
  Do you think trucks are vehicles? Answer: Yes
  ```

- **Auto-discovery**: `discover_rosch_datasets()` scans `data/rosch/` for files matching `rosch-*_test.csv` and registers each as a task.

- **`get_completion()`**: `" " + item.member`
- **`get_label()`**: `item.label` (already "yes"/"no")

---

## Step 5: Register in `src/tasks/__init__.py`

Added: `from . import rosch  # Rosch-1975 category membership eval tasks`

---

## Step 6: Smoke test

```bash
source /u/jdr/venvs/venv_lexcons/bin/activate
python -c "
import sys; sys.path.append('src')
import tasks
from task_registry import is_registered, get_task
print(is_registered('rosch-furniture'))
task = get_task('rosch-furniture')
_, L_test = task['load_data']()
pc = task['make_prompt'](L_test[0], style='generator')
print(pc.prompt, '->', pc.completion)
"
```

---

## What's needed for training data

To create a TRAINING version:
1. Need positive + negative examples (can use `combined_train_categories_final.json` from flora for positives, and the negatives file we generated)
2. Add `generator_sentence` and `discriminator_sentence` to each item in those JSONs (reuse/adapt `add_sentences_to_rosch_jsons.py`)
3. Create a new script analogous to `make_rosch_test_sets.py` that outputs train CSVs
4. Either extend `src/tasks/rosch.py` to handle train splits, or create a new task module
5. Register the training tasks

---

## Key files summary

| File | Purpose |
|------|---------|
| `flora/rosch-1975.json` | Source: Rosch typicality norms with sentences |
| `flora/incorrect_items-to-supplement-rosch-75.json` | Source: extra negatives with sentences |
| `scripts/add_sentences_to_rosch_jsons.py` | One-time: adds sentence fields to JSONs |
| `scripts/make_rosch_test_sets.py` | Generates CSVs from JSONs (pass-through) |
| `data/rosch/rosch-*_test.csv` | Output: per-category test sets |
| `src/tasks/rosch.py` | Task module (auto-discovers CSVs, builds prompts) |
| `src/tasks/__init__.py` | Imports rosch module to trigger registration |
