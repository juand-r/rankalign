"""
Parse rankalign checkpoint folder names into structured metadata.

Folder name format (constructed by ranking_loss_ref.py line 2658):

  v6-{model//--}-delta{delta}-epoch{epoch}--{task}[-with-ref][-all]
    --{direction}--{split}--alpha{x}
    [--tc-neg|--tc-self-step|--tc-self-epoch|--tc-self-online|--tc-self|--tc-online]
    [--lenorm][--single-token-data][--full-completion]
    [--pref{x}][--nllv{x}][--nllg{x}]
    [--force-same-x][--valboost][--vallogodds]
    [--semi{x}|--labelonly{x}]

Key omission defaults (from dashboard_viz_refactor.py lines 236-241):
  - pref   is OMITTED when value == 1.0  → effective default when absent: 1.0
  - nllv   is OMITTED when value == 0.0  → effective default when absent: 0.0
  - nllg   is OMITTED when value == 0.0  → effective default when absent: 0.0
"""

import re
from pathlib import Path


def _extract_float(pattern: str, text: str):
    """Extract a float using a regex with a named group 'weight'. Returns None if not found."""
    m = re.search(pattern, text)
    if m:
        return float(m.group('weight'))
    return None


def _parse_abbreviated(name: str, version: str, original_name: str) -> dict:
    """Parse the abbreviated form produced by ``to_hf_repo_name(prefix='')``.

    Inverse of ``to_hf_repo_name``. Walks the post-``-d<delta>-e<epoch>-``
    tail right-to-left, peeling known flag tokens off the end; whatever
    remains in the middle is the task_segment.

    Flag tokens recognised: tcs/tcn/tco, ln, p<x>, nv<x>, ng<x>, vlo, fsx,
    ppd, cft, sm<x>, lo<x>, fix1.
    """
    m = re.match(rf'^{version}-(.+?)-d(\d+(?:\.\d+)?)-e(\d+)-(.+)$', name)
    if not m:
        raise ValueError(f"Could not parse abbreviated checkpoint name: {name}")

    model_short = m.group(1)         # e.g. "gemma-2-9b-it"
    delta       = float(m.group(2))
    epoch       = int(m.group(3))
    rest        = m.group(4)         # task_segment + flags, all '-' separated

    # Default flag values (match parse_checkpoint_name's omission semantics).
    tc           = None
    lenorm       = False
    pref         = 1.0
    nll_v        = 0.0
    nll_g        = 0.0
    vallogodds   = False
    force_same_x = False
    ppd          = False
    cft          = False
    fix1         = False
    semi         = None
    labelonly    = None

    parts = rest.split('-')
    while parts:
        tok = parts[-1]
        if   tok == 'fix1':                        fix1 = True
        elif tok == 'fsx':                         force_same_x = True
        elif tok == 'ppd':                         ppd = True
        elif tok == 'cft':                         cft = True
        elif tok == 'vlo':                         vallogodds = True
        elif tok == 'ln':                          lenorm = True
        elif tok in ('tcs', 'tcn', 'tco'):
            tc = {'tcs': 'self', 'tcn': 'neg', 'tco': 'online'}[tok]
        elif (mm := re.fullmatch(r'p(\d+(?:\.\d+)?)', tok)):
            pref = float(mm.group(1))
        elif (mm := re.fullmatch(r'nv(\d+(?:\.\d+)?)', tok)):
            nll_v = float(mm.group(1))
        elif (mm := re.fullmatch(r'ng(\d+(?:\.\d+)?)', tok)):
            nll_g = float(mm.group(1))
        elif (mm := re.fullmatch(r'sm(\d+(?:\.\d+)?)', tok)):
            semi = float(mm.group(1))
        elif (mm := re.fullmatch(r'lo(\d+(?:\.\d+)?)', tok)):
            labelonly = float(mm.group(1))
        else:
            break  # not a flag → task_segment ends here
        parts.pop()

    task_segment = '-'.join(parts)

    return {
        'version':       version,
        'model_name':    None,            # org prefix dropped in abbrev form
        'model_short':   model_short,
        'delta':         delta,
        'epoch':         epoch,
        'task_segment':  task_segment,
        'direction':     None,            # not in abbrev form
        'split_type':    None,            # not in abbrev form
        'alpha':         None,            # not in abbrev form
        'tc':            tc,
        'lenorm':        lenorm,
        'pref':          pref,
        'nll_v':         nll_v,
        'nll_g':         nll_g,
        'vallogodds':    vallogodds,
        'force_same_x':  force_same_x,
        'ppd':           ppd,
        'cft':           cft,
        'fix1':          fix1,
        'semi':          semi,
        'labelonly':     labelonly,
        'original_name': original_name,
    }


def parse_checkpoint_name(path: str) -> dict:
    """
    Parse a rankalign checkpoint folder name (or full path) into structured metadata.

    Strips '_merged' suffix before parsing — the metadata is encoded in the base name.

    Accepts three on-disk formats:

      (A) legacy absolute-path-embedded (eval_by_claude pre-fix bug):
            ``v6-_datastor2_jdr_rankalign_models2_v7-google--gemma-...-fix1[_merged]``
          Stripped down to its inner ``v7-...`` portion before parsing.

      (B) un-abbreviated (current default for ≤160-char basenames):
            ``v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-...--fix1[_merged]``

      (C) abbreviated (when raw basename > 160 chars, produced by
          ``src/tasks/common.py:build_model_short`` via ``to_hf_repo_name``):
            ``v7-gemma-2-9b-it-d2.69-e2-membership-...-tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1``

    Returns a dict with keys:
      version, model_name, model_short, delta, epoch, task_segment,
      direction, split_type, alpha, tc, lenorm, pref, nll_v, nll_g,
      vallogodds, force_same_x, ppd, cft, fix1, semi, labelonly, original_name

    For format (C) some fields aren't recoverable from the abbreviated form
    and come back as None: model_name, direction, split_type, alpha.

    Raises ValueError if the name doesn't match any expected format.
    """
    name = Path(path).name
    # Strip "_merged" suffix only — leaves any "--fix1" intact (handled below)
    if name.endswith('_merged'):
        name = name[:-len('_merged')]

    original_name = name

    # Format (A): legacy absolute-path-embedded prefix. Pre-fix
    # `eval_by_claude.py` set `model_short = 'v6-' + modelname.replace('/','_')`
    # so we get e.g. `v6-_datastor2_jdr_rankalign_models2_v7-google--gemma-...`.
    # Strip the wrapper down to the inner `v6-`/`v7-` segment.
    if name.startswith(('v6-_', 'v7-_')):
        inner = re.search(r'_(v[67]-)', name)
        if inner:
            name = name[inner.start() + 1:]

    # Accept either v6- or v7- prefix. v7 is the fix1 generation; v6 is legacy.
    version_match = re.match(r'^(v6|v7)-', name)
    if not version_match:
        raise ValueError(f"Checkpoint name must start with 'v6-' or 'v7-': {name}")
    version = version_match.group(1)

    # Format (C) detection: abbreviated form has `-d<num>-e<num>-` and no
    # `-delta` substring. Dispatch to the abbreviated parser.
    if '-delta' not in name and re.search(r'-d\d+(?:\.\d+)?-e\d+-', name):
        return _parse_abbreviated(name, version, original_name)

    # Extract: model_raw, delta, epoch, rest
    # Format: v{6,7}-{model//--}-delta{delta}-epoch{epoch}--{rest}
    m = re.match(rf'^{version}-(.+?)-delta(\d+(?:\.\d+)?)-epoch(\d+)--(.+)$', name)
    if not m:
        raise ValueError(f"Could not parse checkpoint name: {name}")

    model_raw  = m.group(1)   # e.g. "google--gemma-2-2b"
    delta      = float(m.group(2))
    epoch      = int(m.group(3))
    rest       = m.group(4)   # everything after "--{task}"

    # Reconstruct HF model name and short name
    model_name  = model_raw.replace('--', '/')    # "google/gemma-2-2b"
    model_short = model_name.split('/')[-1]        # "gemma-2-2b"

    # --- Task segment ---
    # Everything between "--{task}" and the first "--{direction}" token.
    # direction is always one of: g2d, d2g, iter, both  (prefixed with --)
    direction_match = re.search(r'--(g2d|d2g|iter|both)(?:--|$)', rest)
    if not direction_match:
        raise ValueError(f"Could not find direction flag (g2d/d2g/iter/both) in: {rest}")

    task_segment = rest[:direction_match.start()]   # e.g. "hypernym-hammers-all"
    # Strip -with-ref if present (not included in HF name per plan)
    task_segment = task_segment.replace('-with-ref', '')

    direction = direction_match.group(1)            # e.g. "d2g"

    # --- Split type ---
    split_match = re.search(r'--(random|hyper|both)(?:--|$)', rest)
    split_type  = split_match.group(1) if split_match else None

    # --- Alpha ---
    alpha_match = re.search(r'--alpha-?(.+?)(?:--|$)', rest)
    alpha = alpha_match.group(1) if alpha_match else None

    # --- Typicality correction (mutually exclusive) ---
    # Order matters: --tc-self is a substring of --tc-self-step / --tc-self-epoch / --tc-self-online.
    if '--tc-neg' in rest:
        tc = 'neg'
    elif '--tc-self-step' in rest:
        tc = 'self'
    elif '--tc-self-epoch' in rest:
        tc = 'self'
    elif '--tc-self-online' in rest:
        tc = 'self'  # legacy (renamed to --tc-self-epoch in training)
    elif '--tc-self' in rest:
        tc = 'self'
    elif '--tc-online' in rest:
        tc = 'online'
    else:
        tc = None

    # --- Boolean flags ---
    lenorm       = '--lenorm' in rest
    vallogodds   = '--vallogodds' in rest
    force_same_x = '--force-same-x' in rest
    # v7 / fix1 generation flags. The order is set by ranking_loss_ref_fix.py
    # and currently looks like: ...--force-same-x--ppd--cft--vallogodds--semi/labelonly--fix1[_merged]
    ppd          = '--ppd' in rest
    cft          = '--cft' in rest
    fix1         = '--fix1' in rest

    # --- Loss weights (apply omission defaults per dashboard_viz_refactor.py) ---
    # pref omitted when == 1.0 → effective default 1.0 when absent
    pref = _extract_float(r'--pref(?P<weight>\d+(?:\.\d+)?)', rest)
    if pref is None:
        pref = 1.0

    # nllv omitted when == 0.0 → effective default 0.0 when absent
    nll_v = _extract_float(r'--nllv(?P<weight>\d+(?:\.\d+)?)', rest)
    if nll_v is None:
        nll_v = 0.0

    # nllg omitted when == 0.0 → effective default 0.0 when absent
    nll_g = _extract_float(r'--nllg(?P<weight>\d+(?:\.\d+)?)', rest)
    if nll_g is None:
        nll_g = 0.0

    # --- Semi-supervised / labeled-only (mutually exclusive) ---
    semi_match      = re.search(r'--semi(\d+(?:\.\d+)?)', rest)
    labelonly_match = re.search(r'--labelonly(\d+(?:\.\d+)?)', rest)
    semi      = float(semi_match.group(1))      if semi_match      else None
    labelonly = float(labelonly_match.group(1)) if labelonly_match else None

    return {
        'version':      version,
        'model_name':   model_name,
        'model_short':  model_short,
        'delta':        delta,
        'epoch':        epoch,
        'task_segment': task_segment,
        'direction':    direction,
        'split_type':   split_type,
        'alpha':        alpha,
        'tc':           tc,
        'lenorm':       lenorm,
        'pref':         pref,
        'nll_v':        nll_v,
        'nll_g':        nll_g,
        'vallogodds':   vallogodds,
        'force_same_x': force_same_x,
        'ppd':          ppd,
        'cft':          cft,
        'fix1':         fix1,
        'semi':         semi,
        'labelonly':    labelonly,
        'original_name': original_name,
    }


# Abbreviated forms for long task names (user-approved short forms)
TASK_ABBREVIATIONS = {
    'hypernym-concat-bananas-to-dogs-double-all': 'hc-b2d-dbl-all',
    'hypernym-concat-bananas-to-dogs-double':     'hc-b2d-dbl',
}


def to_hf_repo_name(parsed: dict, prefix: str = 'rankalign',
                    max_len: int | None = 96) -> str:
    """
    Build a short, unique HuggingFace repo name from parsed checkpoint metadata.
    Default behavior (max_len=96) enforces HF's repo-ID limit. Pass max_len=None
    to relax (used for filename construction in build_model_short).
    Pass prefix='' to omit the leading 'rankalign-' segment.

    Abbreviations used:
      delta{x}      → d{x}
      epoch{n}      → e{n}
      tc-online/self/neg → tco/tcs/tcn
      lenorm        → ln
      pref{x}       → p{x}
      nllv{x}       → nv{x}
      nllg{x}       → ng{x}
      vallogodds    → vlo
      force-same-x  → fsx
      ppd           → ppd      (per-prompt-delta; v7+)
      cft           → cft      (consistency-ft; v7+)
      fix1          → fix1     (kept verbatim to disambiguate v7 fix1 generation)
      semi{x}       → sm{x}
      labelonly{x}  → lo{x}
      Long task names abbreviated via TASK_ABBREVIATIONS map.

    Included: version, model_short, delta, epoch, task_segment,
              tc (if any), lenorm, pref (if != 1.0), nllv (if != 0.0),
              nllg (if != 0.0), vallogodds, force-same-x, ppd, cft,
              semi, labelonly, fix1.

    Dropped:  direction, split_type, alpha, full-completion, with-ref,
              single-token-data, valboost.
    """
    task = TASK_ABBREVIATIONS.get(parsed['task_segment'], parsed['task_segment'])

    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.extend([
        parsed['version'],
        parsed['model_short'],
        f"d{parsed['delta']:g}",
        f"e{parsed['epoch']}",
        task,
    ])

    if parsed.get('tc'):
        tc_map = {'online': 'tco', 'self': 'tcs', 'neg': 'tcn'}
        parts.append(tc_map.get(parsed['tc'], 'tc-' + parsed['tc']))

    if parsed.get('lenorm'):
        parts.append('ln')

    if parsed.get('pref', 1.0) != 1.0:
        parts.append(f"p{parsed['pref']:g}")

    if parsed.get('nll_v', 0.0) != 0.0:
        parts.append(f"nv{parsed['nll_v']:g}")

    if parsed.get('nll_g', 0.0) != 0.0:
        parts.append(f"ng{parsed['nll_g']:g}")

    if parsed.get('vallogodds'):
        parts.append('vlo')

    if parsed.get('force_same_x'):
        parts.append('fsx')

    # v7 / fix1 generation flags. Order: place fsx-tied flags adjacent to fsx
    # (ppd is fsx-only by design), then version/consistency markers at the end.
    if parsed.get('ppd'):
        parts.append('ppd')

    if parsed.get('cft'):
        parts.append('cft')

    if parsed.get('semi') is not None:
        parts.append(f"sm{parsed['semi']:g}")

    if parsed.get('labelonly') is not None:
        parts.append(f"lo{parsed['labelonly']:g}")

    if parsed.get('fix1'):
        parts.append('fix1')

    name = '-'.join(parts)
    if max_len is not None and len(name) > max_len:
        raise ValueError(f"Repo/short name exceeds {max_len} chars ({len(name)}): {name}")
    return name


# ----------------------------------------------------------------------------
# Structural matcher for v7 table builders
# ----------------------------------------------------------------------------
#
# The v7 table-building scripts in scripts/_build_*_table_v7.py used to do
# regex-equality matching against a single string form. With three on-disk
# formats now in play (legacy abs-path A / un-abbreviated B / abbreviated C),
# regexes are brittle. Match structurally instead: parse, then compare fields.

# Sentinel: "this field must NOT be set" (vs `None` which means "unspecified
# in expected — pass through" and a concrete value which means "must equal").
class _MustBeUnset:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self): return "UNSET"

UNSET = _MustBeUnset()


def matches_v7_setting(
    model_short: str,
    *,
    model: str,                 # required: e.g. "gemma-2-9b-it"
    task_segment: str,          # required: e.g. "membership-sans-rosch-v0-all"
    version: str = "v7",
    tc=UNSET,                   # None / 'self' / 'neg' / 'online' / UNSET (must be None)
    lenorm: bool = False,
    pref: float = 1.0,          # default-omitted = 1.0
    nll_v: float = 0.0,         # default-omitted = 0.0
    nll_g: float = 0.0,         # default-omitted = 0.0
    vallogodds: bool = False,
    force_same_x: bool = False,
    ppd: bool = False,
    cft: bool = False,
    fix1: bool = True,          # all v7 ckpts end in --fix1
    semi=UNSET,                 # float or UNSET (must be None) or None (any)
    labelonly=UNSET,            # float or UNSET (must be None) or None (any)
) -> bool:
    """Return True iff the parsed metadata of `model_short` matches the
    structured expected fields. Accepts formats A/B/C transparently.

    Conventions:
      - For boolean and numeric flags, pass the EXPECTED value. Default
        kwargs reflect the implicit "absent flag" semantics of
        parse_checkpoint_name.
      - For tc/semi/labelonly which are None when absent, pass:
          * a concrete value to require it,
          * UNSET (default) to require ABSENCE,
          * the literal None to skip checking.
    """
    try:
        p = parse_checkpoint_name(model_short)
    except ValueError:
        return False

    if p['version']      != version:      return False
    if p['model_short']  != model:        return False
    if p['task_segment'] != task_segment: return False

    if tc is not None:
        if tc is UNSET:
            if p['tc'] is not None: return False
        else:
            if p['tc'] != tc: return False

    if p['lenorm']       != lenorm:       return False
    if p['pref']         != pref:         return False
    if p['nll_v']        != nll_v:        return False
    if p['nll_g']        != nll_g:        return False
    if p['vallogodds']   != vallogodds:   return False
    if p['force_same_x'] != force_same_x: return False
    if p['ppd']          != ppd:          return False
    if p['cft']          != cft:          return False
    if p['fix1']         != fix1:         return False

    if semi is not None:
        if semi is UNSET:
            if p['semi'] is not None: return False
        else:
            if p['semi'] != semi: return False

    if labelonly is not None:
        if labelonly is UNSET:
            if p['labelonly'] is not None: return False
        else:
            if p['labelonly'] != labelonly: return False

    return True


if __name__ == '__main__':
    examples = [
        # Legacy v6 examples (existing behavior preserved)
        ("v6-google--gemma-2-2b-delta0.15-epoch9--hypernym-hammers-all--d2g--random--alpha1.0--tc-online--lenorm--full-completion--nllv1.0--nllg1.0--vallogodds",
         "rankalign-v6-gemma-2-2b-d0.15-e9-hypernym-hammers-all-tco-ln-nv1-ng1-vlo"),
        ("v6-google--gemma-2-2b-delta0.15-epoch2--plausibleqa-all--d2g--random--alpha1.0--tc-self--lenorm--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1",
         "rankalign-v6-gemma-2-2b-d0.15-e2-plausibleqa-all-tcs-ln-p0-nv1-ng1-fsx-sm0.1"),
        ("v6-google--gemma-2-2b-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--tc-self--lenorm--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1",
         "rankalign-v6-gemma-2-2b-d0.15-e2-hc-b2d-dbl-all-tcs-ln-p0-nv1-ng1-fsx-sm0.1"),
        # v7 fix1 examples
        ("v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1_merged",
         "rankalign-v7-gemma-2-9b-it-d2.69-e2-membership-sans-rosch-v0-all-tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1"),
        ("v7-google--gemma-2-2b-it-delta1.74-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
         "rankalign-v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1"),
        # s13: SFT + consistency-ft (cft)
        ("v7-google--gemma-2-2b-it-delta0.5-epoch2--ifeval-concat-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1",
         "rankalign-v7-gemma-2-2b-it-d0.5-e2-ifeval-concat-all-p0-nv1-ng1-cft-lo0.1-fix1"),
    ]
    for name, exp in examples:
        parsed = parse_checkpoint_name(name)
        # v7 examples may exceed the legacy 96-char HF repo cap; use None for
        # the test harness so we can show the full predicted abbrev form.
        result = to_hf_repo_name(parsed, max_len=None)
        status = '✓' if result == exp else '✗'
        print(f"{status} {result}  (len={len(result)})")
        if result != exp:
            print(f"  expected: {exp}")

    # ------------------------------------------------------------------
    # Round-trip tests for the three on-disk formats. For each canonical
    # un-abbreviated form (B), we synthesize the matching abbreviated (C)
    # and legacy abs-path (A) forms and verify they all parse to fields
    # that satisfy matches_v7_setting() against the same expected struct.
    # ------------------------------------------------------------------
    print()
    print("Round-trip tests (formats A / B / C):")
    rt_cases = [
        # (form_B_input, expected kwargs for matches_v7_setting)
        ("v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-sans-rosch-v0-all"
         "--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0"
         "--force-same-x--ppd--vallogodds--semi0.1--fix1_merged",
         dict(model="gemma-2-9b-it", task_segment="membership-sans-rosch-v0-all",
              tc='self', nll_v=1.0, nll_g=1.0, vallogodds=True,
              force_same_x=True, ppd=True, semi=0.1)),
        # s2 (RankAlign): no fsx, no tc, no nll, no vlo
        ("v7-google--gemma-2-2b-it-delta0.85-epoch2--membership-sans-rosch-v0-all"
         "--d2g--random--alpha1.0--full-completion--semi0.1--fix1",
         dict(model="gemma-2-2b-it", task_segment="membership-sans-rosch-v0-all",
              semi=0.1)),
        # s13 SFT + cft: pref=0, nllv=1, nllg=1, cft, labelonly
        ("v7-google--gemma-2-2b-it-delta0.5-epoch2--ifeval-concat-all"
         "--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0"
         "--cft--labelonly0.1--fix1",
         dict(model="gemma-2-2b-it", task_segment="ifeval-concat-all",
              pref=0.0, nll_v=1.0, nll_g=1.0, cft=True, labelonly=0.1)),
    ]
    all_ok = True
    for form_b, expected in rt_cases:
        parsed_b = parse_checkpoint_name(form_b)
        form_c   = to_hf_repo_name(parsed_b, prefix='', max_len=None)
        # Form A: prepend the legacy abs-path-bug wrapper.
        form_a   = "v6-_datastor2_jdr_rankalign_models2_" + form_b

        ok_b = matches_v7_setting(form_b, **expected)
        ok_c = matches_v7_setting(form_c, **expected)
        ok_a = matches_v7_setting(form_a, **expected)
        ok = ok_a and ok_b and ok_c
        all_ok &= ok
        mark = '✓' if ok else '✗'
        print(f"  {mark} A={ok_a} B={ok_b} C={ok_c}  ({form_b[:60]}...)")

    # Negative cases: the same matcher must REJECT a near-miss.
    print()
    print("Negative tests (expected: all reject):")
    neg_input = ("v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-sans-rosch-v0-all"
                 "--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0"
                 "--force-same-x--ppd--vallogodds--semi0.1--fix1_merged")
    # Ask for tc='self' against a tc='neg' name: must fail.
    bad = matches_v7_setting(neg_input, model="gemma-2-9b-it",
                              task_segment="membership-sans-rosch-v0-all",
                              tc='self', nll_v=1.0, nll_g=1.0, vallogodds=True,
                              force_same_x=True, ppd=True, semi=0.1)
    print(f"  {'✓' if not bad else '✗'} tc-self matcher rejects tc-neg name: {not bad}")

    # Ask for fsx=False against fsx=True name: must fail.
    bad2 = matches_v7_setting(neg_input, model="gemma-2-9b-it",
                               task_segment="membership-sans-rosch-v0-all",
                               tc='neg', nll_v=1.0, nll_g=1.0, vallogodds=True,
                               force_same_x=False, ppd=False, semi=0.1)
    print(f"  {'✓' if not bad2 else '✗'} fsx=False matcher rejects fsx=True name: {not bad2}")
