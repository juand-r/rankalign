"""Setting → checkpoint suffix mapping (shared by eval scripts)."""

SETTING_SUFFIXES = {
    "s1": "--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1--fix1",
    "s2": "--full-completion--semi0.1--fix1",
    "s3": "--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
    "s4": "--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
    "s7": "--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
    "s11": "--tc-self--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1",
    "s12": "--tc-neg--full-completion--nllv1.0--nllg1.0--vallogodds--semi0.1--fix1",
}


def get_setting_suffix(setting):
    if setting not in SETTING_SUFFIXES:
        raise ValueError(f"Unknown setting: {setting}. Known: {list(SETTING_SUFFIXES.keys())}")
    return SETTING_SUFFIXES[setting]
