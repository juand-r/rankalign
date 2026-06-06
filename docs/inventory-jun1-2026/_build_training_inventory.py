#!/usr/bin/env python
"""Build the training-jobs inventory (TASK 1) from captured raw source data.

Inputs (in _raw/):
  - models2_listing.txt       v7 local adapter dirs on /datastor2/jdr/rankalign/models2
  - models_v6_listing.txt     v6 local adapter dirs on /datastor2/jdr/rankalign/models
  - hf_repos.json             {org: [{name, private}]} for latkes + TAUR-dev
  - hf_checkpoint_map.json    authoritative 40-entry local-dir -> HF-repo map (partial)

Outputs:
  - _raw/parsed_local.csv, _raw/parsed_hf.csv   (every row parsed, audit trail)
  - v7/training_inventory_v7.md                 (grouped tables, v7 + v7b)
  - v6/training_inventory_v6.md                 (grouped tables, v6)
  - _raw/unparsed.txt                           (anything the parser could not classify)

Setting classification follows docs/inventory-jun1-2026/SETTINGS_REFERENCE.md.
Reproducible: re-run after refreshing _raw/ to regenerate.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "_raw"

# ---- normalization helpers --------------------------------------------------

def norm_model(m: str) -> str:
    m = m.strip("-")
    aliases = {
        "g4-31b": "gemma-4-31B-it",
        "gemma-4-31b-it": "gemma-4-31B-it",
        "gemma2-9b-it": "gemma-2-9b-it",
        "gemma2-2b-it": "gemma-2-2b-it",
        "gemma2-2b": "gemma-2-2b",
    }
    return aliases.get(m, m)


def norm_task(t: str) -> str:
    t = t.strip("-")
    if t.startswith("membership"):
        return "membership"
    if t.startswith("persona"):
        return "persona"
    if t.startswith("ifeval"):
        return "ifeval"
    if "correct-upper" in t or t == "cu":
        return "humaneval-cu"
    if "correct-multi" in t or t == "cm":
        return "humaneval-cm"
    if t.startswith("ambigqa"):
        return "ambigqa"
    if "hc-b2d" in t or "hypernym" in t:
        return "hypernym"
    if "rosch" in t:
        return "rosch"
    return t


def classify_setting(tokens: set[str], explicit_s: str | None) -> str:
    """Return s1..s13 (or a descriptive label) from flag tokens or explicit sN."""
    if explicit_s:
        return explicit_s
    has = lambda *xs: any(x in tokens for x in xs)  # noqa: E731
    pref0 = has("pref0.0", "p0")
    nll = has("nllv1.0", "nv1", "nllg1.0", "ng1")
    cft = has("cft")
    fsx = has("force-same-x", "fsx")
    tcs = has("tc-self", "tcs")
    tcn = has("tc-neg", "tcn")
    if pref0:  # SFT family (always has nll>0)
        return "s13" if cft else "s1"
    if nll:  # comb family
        if fsx:
            return "s4" if tcs else "s7" if tcn else "s3"
        return "s11" if tcs else "s12" if tcn else "comb-notc-nofsx?"
    # pref-only / RankAlign family
    if fsx:
        return "s5" if tcs else "s8" if tcn else "s10"
    return "s6" if tcs else "s9" if tcn else "s2"


SETTING_NAME = {
    "s1": "SFT-lo", "s2": "RankAlign", "s3": "New+fsx", "s4": "New+fsx+tc",
    "s5": "RankAlign+fsx+tc", "s6": "RankAlign+tc", "s7": "New+fsx+negtc",
    "s8": "RankAlign+fsx+negtc", "s9": "RankAlign+negtc", "s10": "RankAlign+fsx",
    "s11": "New+tc", "s12": "New+negtc", "s13": "SFT+cft",
}

# ---- parsers ----------------------------------------------------------------

LOCAL_RE = re.compile(
    r"^(?P<ver>v6|v7)-google--(?P<model>.+?)-delta(?P<delta>[0-9.]+)-epoch(?P<epoch>\d+)--(?P<task>.+?)-all--d2g--random--alpha1\.0(?P<suffix>.*)$"
)


def parse_local(name: str) -> dict | None:
    m = LOCAL_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    tokens = set(t for t in d["suffix"].split("--") if t)
    setting = classify_setting(tokens, None)
    return {
        "source": "local",
        "version": d["ver"],
        "model": norm_model(d["model"]),
        "task": norm_task(d["task"]),
        "setting": setting,
        "epoch": int(d["epoch"]),
        "delta": d["delta"],
        # v7 paper models live on /datastor2/.../models2; the canonical v6 archive (1319 dirs)
        # is on /datastor1/.../models (/datastor2/.../models is only a ~201-dir partial copy).
        "location": ("/datastor2/jdr/rankalign/models2" if d["ver"] == "v7"
                     else "/datastor1/jdr/gv-gap/rankalign/models"),
        "name": name,
        "private": "",
    }


# HF scheme A (flag suffix):  rankalign-<ver>-<model>-d<delta>-e<N>-<task>-all-<flags>
HF_FLAG_RE = re.compile(
    r"^rankalign-(?P<ver>v7b|v7|v6)-(?P<model>.+?)-d(?P<delta>[0-9.]+)-e(?P<epoch>\d+)-(?P<task>.+?)-all-(?P<suffix>.*)$"
)
# HF scheme B (s-number):    rankalign-<ver>-<model>-<task>-s<N>-ep<N>
HF_S_RE = re.compile(
    r"^rankalign-(?P<ver>v7b|v7|v6)-(?P<model>.+?)-(?P<task>membership|persona|ifeval|cu|cm)-s(?P<snum>\d+)-ep(?P<epoch>\d+)$"
)
# correct-multi special:     rankalign-v7-gemma4-31b-correct-multi-s7-new-fsx-negtc (no epoch)
HF_CM_RE = re.compile(r"^rankalign-(?P<ver>v7)-(?P<model>gemma4-31b)-correct-multi-(?P<rest>.+)$")


def parse_hf(name: str, private: bool | None) -> dict | None:
    base = name.split("/")[-1]
    ver = model = task = setting = delta = None
    epoch = -1
    ms = HF_S_RE.match(base)
    if ms:
        d = ms.groupdict()
        ver, model, task = d["ver"], norm_model(d["model"]), norm_task(d["task"])
        setting = "s" + d["snum"]
        epoch = int(d["epoch"])
    else:
        mf = HF_FLAG_RE.match(base)
        if mf:
            d = mf.groupdict()
            ver, model, task = d["ver"], norm_model(d["model"]), norm_task(d["task"])
            delta = d["delta"]
            epoch = int(d["epoch"])
            tokens = set(t for t in d["suffix"].split("-") if t)
            setting = classify_setting(tokens, None)
        else:
            mc = HF_CM_RE.match(base)
            if mc:
                d = mc.groupdict()
                ver, model, task = d["ver"], norm_model(d["model"]), "humaneval-cm"
                tokens = set(d["rest"].split("-"))
                setting = classify_setting(tokens, None)
            else:
                return None
    return {
        "source": "hf",
        "version": ver,
        "model": model,
        "task": task,
        "setting": setting,
        "epoch": epoch,
        "delta": delta or "",
        "location": name.split("/")[0] if "/" in name else "?",
        "name": name,
        "private": "private" if private else ("public" if private is False else "?"),
    }


# ---- run --------------------------------------------------------------------

def main() -> None:
    rows: list[dict] = []
    unparsed: list[str] = []

    # v7 from /datastor2 models2; v6 from the FULL /datastor1 archive (datastor1_models_listing.txt,
    # 1319 dirs) — supersedes the partial /datastor2 models_v6_listing.txt (201).
    for fn in ["models2_listing.txt", "datastor1_models_listing.txt"]:
        for line in (RAW / fn).read_text().splitlines():
            line = line.strip()
            if not line or not line.startswith(("v6-", "v7-")):
                continue
            r = parse_local(line)
            (rows.append(r) if r else unparsed.append(f"local:{line}"))

    hf = json.loads((RAW / "hf_repos.json").read_text())
    for org, repos in hf.items():
        for rec in repos:
            r = parse_hf(rec["name"], rec.get("private"))
            (rows.append(r) if r else unparsed.append(f"hf:{rec['name']} (private={rec.get('private')})"))

    # audit CSVs
    with (RAW / "parsed_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "version", "model", "task", "setting",
                                          "epoch", "delta", "location", "private", "name"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["version"], x["model"], x["task"], x["setting"], x["epoch"])):
            w.writerow(r)
    (RAW / "unparsed.txt").write_text("\n".join(sorted(unparsed)) + "\n")

    # group: (version-family, model, task, setting) -> {epoch: set(locations)}
    # version family: treat v7 and v7b together as "v7*"; v6 separate.
    def vfam(v: str) -> str:
        return "v6" if v == "v6" else "v7"

    grp: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    # grp[vfam][(model,task,setting)][epoch] = set("local","latkes","TAUR-dev")
    delta_seen: dict = defaultdict(set)
    for r in rows:
        loc = "local" if r["source"] == "local" else r["location"]
        key = (r["model"], r["task"], r["setting"])
        grp[vfam(r["version"])][key][r["epoch"]].add(loc)
        if r["delta"]:
            delta_seen[(vfam(r["version"]),) + key].add(r["delta"])

    def render(vf: str) -> str:
        out = [f"# Training inventory — {vf}{'*  (v7 + v7b)' if vf=='v7' else ''}",
               "",
               "Auto-generated by `_build_training_inventory.py` from `_raw/` source data + HF repo "
               "listings. `L` = the local on-disk copy; for **v7** that is "
               "`/datastor2/jdr/rankalign/models2`, for **v6** that is "
               "`/datastor1/jdr/gv-gap/rankalign/models` (the canonical archive). "
               "`latkes`/`TAUR` = HuggingFace. Empty = checkpoint not found.",
               "",
               "See `SETTINGS_REFERENCE.md` for what each setting means.",
               "",
               "> **datastor1 second pass DONE (2026-06-02).** datastor1 is back. Findings: "
               "`/datastor1/.../models` holds the full **v6** archive (1319 dirs, all v6 — 0 v7, "
               "0 gemma-4, 0 qwen), so the v6 table below is now built from it (the "
               "`/datastor2/.../models` copy was only ~201 dirs, a partial subset). **v7/paper "
               "models are NOT on datastor1** — they live on `/datastor2/.../models2` + HF, so the "
               "v7 inventory was already complete.",
               ""]
        tasks = sorted({k[1] for k in grp[vf]})
        for task in tasks:
            out.append(f"\n## {task}\n")
            out.append("| Model | Setting | Name | e0 | e1 | e2 | delta(s) |")
            out.append("|---|---|---|---|---|---|---|")
            keys = sorted([k for k in grp[vf] if k[1] == task],
                          key=lambda k: (k[0], _skey(k[2])))
            for (model, _task, setting) in keys:
                epd = grp[vf][(model, task, setting)]
                cells = []
                for e in (0, 1, 2):
                    locs = epd.get(e, set())
                    if not locs:
                        cells.append("·")
                    else:
                        tag = []
                        if "local" in locs:
                            tag.append("L")
                        if "latkes" in locs:
                            tag.append("latkes")
                        if "TAUR-dev" in locs:
                            tag.append("TAUR")
                        cells.append("+".join(tag))
                deltas = sorted(delta_seen.get((vf, model, task, setting), set()))
                dstr = ",".join(deltas[:6]) + ("…" if len(deltas) > 6 else "")
                sname = SETTING_NAME.get(setting, setting)
                out.append(f"| {model} | {setting} ({sname}) | | {cells[0]} | {cells[1]} | {cells[2]} | {dstr} |")
        return "\n".join(out) + "\n"

    (HERE / "v7" / "training_inventory_v7.md").write_text(render("v7"))
    (HERE / "v6" / "training_inventory_v6.md").write_text(render("v6"))

    print(f"parsed {len(rows)} rows; {len(unparsed)} unparsed (see _raw/unparsed.txt)")
    print("v7 cells:", sum(len(v) for k, v in grp["v7"].items()) if False else len(grp["v7"]))
    print("v6 cells:", len(grp["v6"]))


def _skey(s: str):
    m = re.match(r"s(\d+)", s)
    return (0, int(m.group(1))) if m else (1, s)


if __name__ == "__main__":
    main()
