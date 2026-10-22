"""Load the YAML configuration, fill defaults, resolve paths and check that the required names are there."""
import copy
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    "run_name": "run",
    "output_dir": "outputs",
    "data": {
        "bank": {"sheet": None, "date_column": "date", "frequency": "Q"},
        "events": {"file": None, "sheet": None, "date_column": "date", "series_column": "series",
                   "amount_column": "amount"},
        "macro": {"date_column": "date", "sheet": None},
        "metadata": {"sheet": None, "columns": {"mnemonic": "mnemonic", "description": "description",
                                                "geography": "geography", "theme": "theme",
                                                "transform": "transform", "expected_sign": "expected_sign"}},
        "constructed": [],
    },
    "sample": {"horizon": 9, "dummy_quarters": []},
    "features": {"max_lag": 4, "adf_p": 0.05, "transforms_level": ["dlog", "yoy", "dlog2", "acc", "gap"],
                 "transforms_rate": ["diff", "diff2", "diff4", "lvl", "rgap"], "real_themes": [],
                 "deflator": None, "allowed_geographies": [], "excluded_themes": [], "require_sign_prior": True},
    "approaches": ["screen_first", "en_first"],
    "screen_first": {"screen_p": 0.05, "cluster_corr": 0.75, "primary_geography": None, "rep_tolerance": 0.75,
                     "shortlist_max": 25},
    "en_first": {"l1_ratio": 0.5, "target_n": 10, "unpenalized_controls": True, "sign_constraints": True,
                 "stability": {"enabled": True, "n_subsamples": 100, "fraction": 0.75, "block_length": 8,
                               "threshold": 0.6, "seed": 42},
                 "group_corr": 0.75, "primary_geography": None, "rep_tolerance": 0.75},
    "cache": {"enabled": True, "dir": "cache"},
    "final_model": {"min_vars": 1, "max_vars": 3, "t_min": 1.96, "vif_max": 5.0, "one_per_theme": True,
                    "required_themes": [], "n_oos_candidates": 300, "selection_tolerance": 0.05},
}

REQUIRED = [
    ("data", "bank", "file"), ("data", "bank", "target_column"),
    ("data", "macro", "scenarios"), ("data", "macro", "base_scenario"),
    ("data", "metadata", "file"),
    ("sample", "hist_end"), ("sample", "estimation_start"), ("sample", "backtest_start"),
]


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _path(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def load_config(path):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    missing = []
    for keys in REQUIRED:
        node = cfg
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node in (None, "", {}):
            missing.append(".".join(keys))
    if missing:
        raise ValueError(f"Config {path} is missing: {', '.join(missing)}")
    d = cfg["data"]
    d["bank"]["file"] = _path(d["bank"]["file"])
    d["events"]["file"] = _path(d["events"]["file"])
    d["metadata"]["file"] = _path(d["metadata"]["file"])
    d["macro"]["scenarios"] = {k: _path(v) for k, v in d["macro"]["scenarios"].items()}
    if d["macro"]["base_scenario"] not in d["macro"]["scenarios"]:
        raise ValueError(f"base_scenario '{d['macro']['base_scenario']}' is not one of the scenarios")
    for label, f in [("bank", d["bank"]["file"]), ("metadata", d["metadata"]["file"])] + \
            [(f"scenario {k}", v) for k, v in d["macro"]["scenarios"].items()]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{label} file not found: {f}")
    for k in ("hist_end", "estimation_start", "backtest_start"):
        cfg["sample"][k] = str(cfg["sample"][k])
    cfg["sample"]["dummy_quarters"] = [str(q) for q in cfg["sample"]["dummy_quarters"]]
    cfg["out_dir"] = os.path.join(_path(cfg["output_dir"]), cfg["run_name"])
    cfg["cache"]["dir"] = _path(cfg["cache"]["dir"])
    cfg["config_file"] = os.path.abspath(path)
    return cfg
