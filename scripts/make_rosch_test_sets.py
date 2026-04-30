"""
Generate per-category test sets for the Rosch-1975 typicality task.

Sources:
  - Positives: flora/rosch-1975.json (items where correct == "yes")
  - Negatives: flora/rosch-1975.json (items where correct == "no")
              + flora/incorrect_items-to-supplement-rosch-75.json (extra negatives)

Output: data/rosch/<task_name>_test.csv for each category, where
  task_name = "rosch-<category>" with spaces replaced by hyphens.

CSV schema: category, member, label, generator_sentence, discriminator_sentence,
            rank, similarity_score, distant
"""

import csv
import json
from pathlib import Path

FLORA_DIR = Path(__file__).resolve().parent.parent.parent / "flora"
ROSCH_FILE = FLORA_DIR / "rosch-1975.json"
SUPPLEMENT_FILE = FLORA_DIR / "incorrect_items-to-supplement-rosch-75.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "rosch"

FIELDNAMES = [
    "category", "member", "label",
    "generator_sentence", "discriminator_sentence",
    "rank", "similarity_score", "distant",
]


def category_to_task_slug(category_name: str) -> str:
    return category_name.lower().replace("'s ", "s-").replace(" ", "-")


def main():
    with open(ROSCH_FILE) as f:
        rosch = json.load(f)

    with open(SUPPLEMENT_FILE) as f:
        supplement = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cat_name, items in rosch["categories"].items():
        slug = category_to_task_slug(cat_name)
        task_name = f"rosch-{slug}"
        cat_lower = cat_name.lower()

        rows = []

        for item in items:
            label = "yes" if item["correct"] == "yes" else "no"
            rows.append({
                "category": cat_lower,
                "member": item["member"],
                "label": label,
                "generator_sentence": item["generator_sentence"],
                "discriminator_sentence": item["discriminator_sentence"],
                "rank": item.get("rank", ""),
                "similarity_score": item.get("specific_score", ""),
                "distant": "",
            })

        supplement_items = supplement.get("categories", {}).get(cat_name, [])
        for item in supplement_items:
            rows.append({
                "category": cat_lower,
                "member": item["member"],
                "label": "no",
                "generator_sentence": item["generator_sentence"],
                "discriminator_sentence": item["discriminator_sentence"],
                "rank": "",
                "similarity_score": "",
                "distant": item.get("distant", ""),
            })

        out_path = OUTPUT_DIR / f"{task_name}_test.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        n_pos = sum(1 for r in rows if r["label"] == "yes")
        n_neg = sum(1 for r in rows if r["label"] == "no")
        print(f"  {task_name}: {n_pos} pos + {n_neg} neg = {len(rows)} total -> {out_path.name}")

    print(f"\nAll test sets written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
