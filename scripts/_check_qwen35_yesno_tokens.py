"""One-off: confirm Qwen3.5 tokenizers handle our yes/no word list cleanly.

We need each of {"Yes", " Yes", "YES", "yes", " yes"} (and the No variants)
to encode to a single token, and we need the resulting yestoks/notoks ID
sets to be distinct (so log-odds isn't comparing overlapping IDs).

Verifies for Qwen/Qwen3.5-{4B, 9B} (the 27B uses the same tokenizer family
but checked separately if asked).
"""

from transformers import AutoTokenizer

YES_WORDS = ["Yes", " Yes", "YES", "yes", " yes"]
NO_WORDS = ["No", " No", "NO", "no", " no"]

MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]


def check_one(model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"  FAILED to load tokenizer: {e}")
        return

    print(f"  bos_token_id = {tok.bos_token_id}")
    print(f"  eos_token_id = {tok.eos_token_id}")
    print(f"  vocab size   = {len(tok)}")

    def report(words, label):
        print(f"\n  {label}:")
        last_ids = []
        for w in words:
            ids = tok.encode(w)
            ids_no_special = tok.encode(w, add_special_tokens=False)
            decoded = [tok.decode([i]) for i in ids]
            last_ids.append(ids[-1])
            single = "single" if len(ids_no_special) == 1 else f"MULTI ({len(ids_no_special)})"
            print(
                f"    {w!r:>10}  encode={ids}  no_special={ids_no_special}"
                f"  {single}  decoded={decoded}"
            )
        return last_ids

    yestoks = report(YES_WORDS, "yes_words")
    notoks = report(NO_WORDS, "no_words")

    print(f"\n  yestoks (last-token-of-each) = {yestoks}")
    print(f"  notoks  (last-token-of-each) = {notoks}")
    print(f"  unique yestoks: {sorted(set(yestoks))}")
    print(f"  unique notoks : {sorted(set(notoks))}")

    overlap = set(yestoks) & set(notoks)
    if overlap:
        print(f"  !!! OVERLAP between yes and no: {overlap}")
    else:
        print("  OK: no overlap between yes-IDs and no-IDs")


for m in MODELS:
    check_one(m)
