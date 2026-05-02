# Dataset Builder

Modular framework for generating and building rankalign task datasets. Each task (GSM8K, HumanEval, etc.) needs positive (correct) and negative (incorrect) solutions for the same set of problems. This framework handles the full pipeline: generate solutions via LLM, validate correctness, and build balanced CSV datasets.

## Directory Structure

```
scripts/dataset_builder/
├── README.md                          # This file
├── generate_solutions.py              # Unified CLI for generation + stats + examples
├── build_gsm8k_dataset.py             # GSM8K-specific dataset builder (full + truncated versions)
├── build_humaneval_dataset.py          # HumanEval dataset builder (balanced CSVs)
└── solution_generator/                 # Core library
    ├── __init__.py
    ├── base.py                         # TaskConfig ABC, GenerationConfig, generate/stats/examples
    ├── gsm8k_config.py                 # GSM8K: prompts, answer extraction, validation
    └── humaneval_config.py             # HumanEval: prompts, code execution, test validation
```

## Quick Start

All commands run from the repo root.

### 1. Generate solutions

```bash
# GSM8K: generate 20 solutions per problem using gpt-4o-mini
python scripts/dataset_builder/generate_solutions.py gsm8k \
    --problems data/gsm8k/gsm8k_train_problems.jsonl \
    --output data/gsm8k/solutions.jsonl \
    --model gpt-4o-mini --samples 20

# Generate intentionally wrong solutions (for problems that need more negatives)
python scripts/dataset_builder/generate_solutions.py gsm8k \
    --problems data/gsm8k/gsm8k_train_problems.jsonl \
    --output data/gsm8k/solutions.jsonl \
    --strategy intentional_bug --model gpt-4o --target neg

# HumanEval
python scripts/dataset_builder/generate_solutions.py humaneval \
    --problems data/humaneval/problems.jsonl \
    --output data/humaneval/solutions.jsonl
```

### 2. Check stats and examples

```bash
# Stats
python scripts/dataset_builder/generate_solutions.py gsm8k \
    --stats --output data/gsm8k/solutions.jsonl

# Show 5 correct and 5 incorrect examples
python scripts/dataset_builder/generate_solutions.py gsm8k \
    --examples --n-examples 5 --output data/gsm8k/solutions.jsonl
```

### 3. Build dataset CSVs

```bash
# GSM8K (creates full_response and truncated_response versions)
python scripts/dataset_builder/build_gsm8k_dataset.py --balance --max-per-side 30

# HumanEval
python scripts/dataset_builder/build_humaneval_dataset.py --balance --train-count 80
```

## How It Works

### Solution JSONL Format

All generators write to the same flat JSONL format (one solution per line):

```json
{
    "question_id": "gsm8k_train_42",
    "question": "A farmer has 24 apples...",
    "response": "Step 1: The farmer has 24 apples...\nThe answer is 12.",
    "raw_response": "...(original LLM output before cleaning)...",
    "correct": true,
    "extracted_answer": "12",
    "gold_answer": "12",
    "error": null,
    "temperature": 1.0,
    "model": "gpt-4o-mini",
    "strategy": "normal",
    "task": "gsm8k"
}
```

Key fields:
- `correct`: boolean, determined by task-specific validation
- `strategy`: `"normal"` (standard sampling) or `"intentional_bug"` (prompted to make mistakes)
- `extracted_answer`: what the validator pulled from the response (for debugging)
- `error`: error message if validation failed, null otherwise

### Dataset CSV Format

The build scripts convert JSONL into the rankalign CSV format:

```
question,answer,correct,strategy
"A farmer has 24 apples...","Step 1: ...The answer is 12.",Yes,gsm8k
```

Columns: `question`, `answer`, `correct` (Yes/No), `strategy`.

These CSVs go into `data/<task>/with_solutions/` and are read by the task loaders in `src/tasks/`.

## Adding a New Task

To add a new task (e.g., k-sat, logic puzzles), you need two things:

### Step 1: Create a TaskConfig subclass

Create `scripts/dataset_builder/solution_generator/<task>_config.py`:

```python
from .base import TaskConfig

class MyTaskConfig(TaskConfig):
    def __init__(self):
        super().__init__(name="mytask")

    def make_prompt(self, problem: dict, strategy: str = "normal") -> list[dict]:
        """Return OpenAI chat messages for generating a solution.

        Args:
            problem: dict with 'question_id', 'question', 'gold_answer',
                     and any task-specific fields
            strategy: "normal" or "intentional_bug"

        Returns: list of {"role": ..., "content": ...} dicts
        """
        if strategy == "intentional_bug":
            return [
                {"role": "system", "content": "You are a student who makes subtle mistakes..."},
                {"role": "user", "content": problem['question']},
            ]
        else:
            return [
                {"role": "system", "content": "Solve this problem step by step..."},
                {"role": "user", "content": problem['question']},
            ]

    def validate(self, problem: dict, raw_solution: str) -> tuple[bool, str, str | None]:
        """Check if a solution is correct.

        Returns: (passed, extracted_answer, error_message_or_None)

        For math: extract final number, compare to gold.
        For code: run tests.
        For logic: check satisfiability.
        For classification: compare label.
        """
        extracted = extract_answer(raw_solution)  # your extraction logic
        gold = problem['gold_answer']
        passed = (extracted == gold)
        return passed, extracted, None if passed else f"expected={gold}, got={extracted}"

    def clean_solution(self, raw_solution: str) -> str:
        """Optional: post-process raw LLM output. Default: strip()."""
        return raw_solution.strip()
```

### Step 2: Register it in generate_solutions.py

Add to the `TASK_CONFIGS` and `DEFAULT_OUTPUTS` dicts in `generate_solutions.py`:

```python
TASK_CONFIGS = {
    'gsm8k': 'solution_generator.gsm8k_config:GSM8KConfig',
    'humaneval': 'solution_generator.humaneval_config:HumanEvalConfig',
    'mytask': 'solution_generator.mytask_config:MyTaskConfig',  # <-- add this
}

DEFAULT_OUTPUTS = {
    'gsm8k': 'data/gsm8k/solutions.jsonl',
    'humaneval': 'data/humaneval/solutions.jsonl',
    'mytask': 'data/mytask/solutions.jsonl',  # <-- add this
}
```

### Step 3: Prepare problems JSONL

Create a JSONL file with one problem per line. Required fields:

```json
{"question_id": "mytask_0", "question": "...", "gold_answer": "..."}
```

Add any extra fields your `validate()` needs (e.g., `test_code` for HumanEval, `entry_point`, etc.).

### Step 4: Generate and build

```bash
# Generate solutions
python scripts/dataset_builder/generate_solutions.py mytask \
    --problems data/mytask/problems.jsonl --samples 20

# Check stats
python scripts/dataset_builder/generate_solutions.py mytask --stats

# If not enough negatives, generate intentional bugs
python scripts/dataset_builder/generate_solutions.py mytask \
    --problems data/mytask/problems.jsonl \
    --strategy intentional_bug --model gpt-4o --target neg

# Write a build_mytask_dataset.py if you need special processing
# (or use the JSONL directly if the standard format is fine)
```

### Step 5 (optional): Write a dataset builder

If your task needs special processing (like GSM8K's truncated version or HumanEval's OOD split), write a `build_<task>_dataset.py` script. Otherwise, the raw JSONL + a simple CSV converter is sufficient.

## Generation Strategies

### `normal` (default)
Standard sampling from the LLM. Good models will produce mostly correct solutions. Adjust temperature to control diversity:
- Low temp (0.2-0.6): more correct, less diverse
- High temp (1.0-1.4): more errors, more diverse

### `intentional_bug`
Prompts the LLM to deliberately introduce subtle mistakes. Much higher failure rate (~35-45%) than normal sampling. Use this when normal sampling doesn't produce enough incorrect solutions.

### Targeting
Use `--target neg` to only generate for problems that still need negatives (below `--threshold`). Similarly `--target pos` for positives. The generator automatically skips problems that already have enough solutions (reads the existing output JSONL).

### Multi-round workflow
Typical workflow for a new task:

1. `--strategy normal --model gpt-4o-mini --samples 20` — initial round
2. Check stats. If too few negatives:
3. `--strategy intentional_bug --model gpt-4o --target neg --samples 20` — targeted bug generation
4. Repeat until enough problems qualify (>=10 correct AND >=10 incorrect)

## Existing Task Configs

### GSM8K (`gsm8k_config.py`)
- **Validation:** Extract final number from response, compare to gold answer
- **Answer extraction:** Tries `#### N`, `The answer is N`, `\boxed{N}`, last-number fallback
- **Normal prompt:** "Solve step by step... end with 'The answer is <number>.'"
- **Bug prompt:** "You are a student who makes subtle calculation errors..."

### HumanEval (`humaneval_config.py`)
- **Validation:** Run generated code against HumanEval unit tests in subprocess
- **Clean:** Strip markdown code blocks, extract function body, fix indentation
- **Normal prompt:** "Complete the function body only, no signature/imports/explanation"
- **Bug prompt:** "Introduce a subtle bug — off-by-one, wrong operator, missing edge case"

## Using Pre-existing Datasets

Some tasks have existing datasets with labeled solutions (correct/incorrect) that can be downloaded from HuggingFace instead of generated:

- **GSM8K test:** [RLHFlow/Mistral-GSM8K-Test](https://huggingface.co/datasets/RLHFlow/Mistral-GSM8K-Test) — 1,319 problems x 1,024 solutions, binary labels. Mistral-7B generations.
- **GSM8K train (correct only):** [hkust-nlp/dart-math-pool-gsm8k](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-gsm8k) — 2.7M correct solutions, no negatives.

Downloaded data should be converted to the standard JSONL format (see above) before using with the build scripts.
