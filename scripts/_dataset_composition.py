#!/usr/bin/env python3
"""Extract training-set composition stats per task.

For each task, computes from the **train side** of `load_data(seed=0,
split_type='random')`:

- Total train items (with `--all` semantics: includes labeled negatives)
- Distinct generator prompts (the key used by `--force-same-x` in g-mode)
- Per-prompt: n_items, n_pos, n_neg

Emits a single markdown document. Per workspace rule, every number in the
output comes from this script's stdout (no hand-typing).

Usage:
  source /u/jdr/venvs/venv_lexcons/bin/activate
  python scripts/_dataset_composition.py [task1 task2 ...] [--out FILE]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task_registry import get_task, get_all_task_names  # noqa: E402

# Trigger registrations (mirrors src/tasks/__init__.py). Registry/import
# code prints to stdout; redirect to stderr so the emitted markdown stays clean.
import contextlib  # noqa: E402

with contextlib.redirect_stdout(sys.stderr):
    import tasks  # noqa: F401, E402

# Tasks we care about for the paper. Override on CLI.
DEFAULT_TASKS = [
    "persona-v1",
    "membership-sans-rosch-v0",
    "humaneval-v2.1correct-multi",
    "humaneval-v2.1correct-upper",
    "ifeval-concat",
]


def task_stats(name: str):
    """Returns (by_prompt: dict[str -> {'total','yes','no'}], n_total: int) or None."""
    cfg = get_task(name)
    if cfg is None:
        return None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            L_train, _ = cfg["load_data"](seed=0, split_type="random")
    except Exception as exc:
        print(f"# WARN: failed to load {name}: {exc}", file=sys.stderr)
        return None

    by_prompt: dict = defaultdict(lambda: {"total": 0, "yes": 0, "no": 0})
    for item in L_train:
        try:
            pc = cfg["make_prompt"](
                item, style="generator", shots="zero", neg=False
            )
            prompt = pc.prompt
        except Exception as exc:
            print(f"# WARN: make_prompt failed for {name}: {exc}", file=sys.stderr)
            continue
        try:
            label = cfg["get_label"](item)
        except Exception as exc:
            print(f"# WARN: get_label failed for {name}: {exc}", file=sys.stderr)
            continue

        by_prompt[prompt]["total"] += 1
        if str(label).lower() == "yes":
            by_prompt[prompt]["yes"] += 1
        elif str(label).lower() == "no":
            by_prompt[prompt]["no"] += 1
        # else: leave uncounted in pos/neg, but counted in total

    return by_prompt, len(L_train)


def ifeval_concat_split_stats():
    """Return dict with ID-train, ID-test, OOD-test breakdowns for ifeval-concat.

    Mirrors the routing logic in src/tasks/ifeval_concat.py:
      - prompts named prompt_1 through prompt_21: OOD, all items go to test
      - all other prompts: ID, 50/50 train/test split

    For each bucket, returns: total items, distinct prompts, pos, neg.
    """
    cfg = get_task("ifeval-concat")
    if cfg is None:
        return None
    # Reuse load_data so train/test routing matches exactly. Then re-derive
    # which prompts are OOD vs ID by inspecting items in each bucket.
    with contextlib.redirect_stdout(sys.stderr):
        L_train, L_test = cfg["load_data"](seed=0, split_type="random")

    # Lazy import to access the task module's helpers.
    from tasks import ifeval_concat as ifc

    def bucket(items):
        by_prompt = defaultdict(lambda: {"total": 0, "yes": 0, "no": 0})
        for item in items:
            pc = cfg["make_prompt"](
                item, style="generator", shots="zero", neg=False
            )
            label = str(cfg["get_label"](item)).lower()
            d = by_prompt[pc.prompt]
            d["total"] += 1
            if label == "yes":
                d["yes"] += 1
            elif label == "no":
                d["no"] += 1
            d["_raw_prompt_name"] = item.get("dataset_source") or \
                item.get("prompt_name") or ""
        return by_prompt

    # Identify each item's source prompt by re-loading raw per-prompt data
    # and matching items. Cheaper: just re-run discover + per-prompt load and
    # bucket items there.
    prompts = ifc.discover_ifeval_datasets()
    id_train = defaultdict(lambda: {"total": 0, "yes": 0, "no": 0})
    id_test = defaultdict(lambda: {"total": 0, "yes": 0, "no": 0})
    ood_test = defaultdict(lambda: {"total": 0, "yes": 0, "no": 0})

    import math as _math
    import utils as _utils  # noqa: F401  (already on sys.path via repo src)
    for prompt_name in prompts:
        dataset = ifc.load_ifeval_data_raw(prompt_name)
        if not dataset:
            continue
        if ifc.is_test_only_prompt(prompt_name):
            target = ood_test[prompt_name]
            for item in dataset:
                target["total"] += 1
                lab = str(cfg["get_label"](item)).lower()
                if lab == "yes":
                    target["yes"] += 1
                elif lab == "no":
                    target["no"] += 1
        else:
            num_train = _math.floor(len(dataset) * 0.5)
            train_items, test_items = _utils.split_train_test(
                dataset, seed=ifc.SEED, subsample=False, num_train=num_train
            )
            if not train_items or not test_items:
                continue
            for item in train_items:
                d = id_train[prompt_name]
                d["total"] += 1
                lab = str(cfg["get_label"](item)).lower()
                if lab == "yes":
                    d["yes"] += 1
                elif lab == "no":
                    d["no"] += 1
            for item in test_items:
                d = id_test[prompt_name]
                d["total"] += 1
                lab = str(cfg["get_label"](item)).lower()
                if lab == "yes":
                    d["yes"] += 1
                elif lab == "no":
                    d["no"] += 1

    return {
        "id_train": id_train,
        "id_test": id_test,
        "ood_test": ood_test,
        "n_train_total": len(L_train),
        "n_test_total": len(L_test),
    }


def render_ifeval_split_section(stats):
    """Markdown lines for the ifeval-concat OOD/ID breakdown."""
    out = []
    out.append("### `ifeval-concat`: ID / OOD breakdown\n")
    out.append(
        "ifeval-concat routes prompts named `prompt_1` through `prompt_21` "
        "entirely into the test set (OOD), and 50/50-splits all other "
        "prompts between train and test (ID). The train-side row in the "
        "summary table above is therefore **ID-only by construction**.\n"
    )

    def summarize(name, by_prompt):
        n_prompts = len(by_prompt)
        n_total = sum(p["total"] for p in by_prompt.values())
        n_pos = sum(p["yes"] for p in by_prompt.values())
        n_neg = sum(p["no"] for p in by_prompt.values())
        denom = n_pos + n_neg
        frac = (n_pos / denom) if denom > 0 else float("nan")
        frac_str = "n/a" if denom == 0 else f"{frac:.3f}"
        return name, n_total, n_prompts, n_pos, n_neg, frac_str

    rows = [
        summarize("ID-train", stats["id_train"]),
        summarize("ID-test",  stats["id_test"]),
        summarize("OOD-test", stats["ood_test"]),
    ]
    out.append("| Split | Items | Distinct prompts | Pos | Neg | Pos/(Pos+Neg) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, n_total, n_prompts, n_pos, n_neg, frac_str in rows:
        out.append(
            f"| {name} | {n_total} | {n_prompts} | {n_pos} | {n_neg} | {frac_str} |"
        )
    out.append("")

    # Sanity totals.
    out.append(
        f"Cross-check: ID-train + ID-test + OOD-test items = "
        f"{rows[0][1] + rows[1][1] + rows[2][1]}; "
        f"L_train (load_data) = {stats['n_train_total']}, "
        f"L_test (load_data) = {stats['n_test_total']}.\n"
    )

    # OOD per-prompt table (small, all 21).
    out.append("**OOD-test per-prompt** (held-out, no train items):\n")
    out.append("| Prompt name | Items | Pos | Neg |")
    out.append("|---|---:|---:|---:|")
    for pname in sorted(stats["ood_test"].keys(),
                        key=lambda s: int(s.split("_")[1]) if s.startswith("prompt_") and s.split("_")[1].isdigit() else 99999):
        d = stats["ood_test"][pname]
        out.append(f"| `{pname}` | {d['total']} | {d['yes']} | {d['no']} |")
    out.append("")
    return out


def fmt_prompt(prompt: str, max_len: int = 100) -> str:
    """Single-line, table-cell-safe prompt, truncated."""
    p = prompt.replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    if len(p) > max_len:
        p = p[: max_len - 3] + "..."
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", default=[])
    parser.add_argument("--out", type=str, default=None,
                        help="Write markdown to this path (default: stdout).")
    args = parser.parse_args()

    task_list = args.tasks if args.tasks else DEFAULT_TASKS

    # Collect.
    per_task = []
    for name in task_list:
        result = task_stats(name)
        if result is None:
            print(f"# SKIP: task {name!r} not in registry or failed to load",
                  file=sys.stderr)
            continue
        by_prompt, n_total = result
        per_task.append((name, by_prompt, n_total))

    # Emit markdown.
    out = []
    out.append("# Training-set composition per task\n")
    out.append(
        "Generated by `scripts/_dataset_composition.py`. For each task this "
        "reports the **train side** of `load_data(seed=0, split_type='random')` "
        "with `--all` semantics (i.e. labeled negatives are kept). The "
        "`Distinct prompts` column is the count of unique **generator** "
        "prompts (the key used by `--force-same-x` in g-mode pair construction).\n"
    )
    cmd = "python scripts/_dataset_composition.py " + " ".join(task_list) + \
          " --out docs/dataset_composition.md"
    out.append("Regenerate with:\n")
    out.append("```bash")
    out.append("source /u/jdr/venvs/venv_lexcons/bin/activate")
    out.append(cmd)
    out.append("```")
    out.append("")

    out.append("## Cross-task summary\n")
    out.append(
        "| Task | Train items | Distinct generator prompts | Pos | Neg | Pos/(Pos+Neg) |"
    )
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, by_prompt, n_total in per_task:
        n_prompts = len(by_prompt)
        n_pos = sum(p["yes"] for p in by_prompt.values())
        n_neg = sum(p["no"] for p in by_prompt.values())
        denom = n_pos + n_neg
        frac = (n_pos / denom) if denom > 0 else float("nan")
        frac_str = "n/a" if denom == 0 else f"{frac:.3f}"
        out.append(
            f"| `{name}` | {n_total} | {n_prompts} | {n_pos} | {n_neg} | {frac_str} |"
        )
    out.append("")

    # Per-task detail.
    for name, by_prompt, n_total in per_task:
        n_prompts = len(by_prompt)
        n_pos = sum(p["yes"] for p in by_prompt.values())
        n_neg = sum(p["no"] for p in by_prompt.values())

        out.append(f"## `{name}`\n")
        out.append(f"- **Total train items**: {n_total}")
        out.append(f"- **Distinct generator prompts**: {n_prompts}")
        out.append(f"- **Pos**: {n_pos} &nbsp;&nbsp;**Neg**: {n_neg}\n")

        if n_prompts == 0:
            out.append("_No items._\n")
            continue

        if n_prompts == 1:
            (prompt, stats), = by_prompt.items()
            out.append(
                "Single generator prompt for this task:\n"
            )
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append("")
            out.append(
                f"`--force-same-x` is therefore a no-op for this task "
                f"(all pairs already share a prompt)."
            )
            out.append("")
            continue

        # Per-prompt table, sorted by total descending.
        rows = sorted(by_prompt.items(), key=lambda kv: -kv[1]["total"])

        # Items-per-prompt summary stats.
        sizes = [v["total"] for _, v in rows]
        n_sizes = len(sizes)
        s_min, s_max = sizes[-1], sizes[0]
        s_mean = sum(sizes) / n_sizes
        s_med = sorted(sizes)[n_sizes // 2]
        out.append(
            f"Items-per-prompt: min={s_min}, median={s_med}, "
            f"mean={s_mean:.1f}, max={s_max}.\n"
        )

        out.append("| # | Prompt (truncated to 100 chars) | Items | Pos | Neg |")
        out.append("|---:|---|---:|---:|---:|")
        for i, (prompt, stats) in enumerate(rows, 1):
            out.append(
                f"| {i} | `{fmt_prompt(prompt)}` | "
                f"{stats['total']} | {stats['yes']} | {stats['no']} |"
            )
        out.append("")

        if name == "ifeval-concat":
            split_stats = ifeval_concat_split_stats()
            if split_stats is not None:
                out.extend(render_ifeval_split_section(split_stats))

    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
