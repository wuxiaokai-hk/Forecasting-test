"""Hash-based caching of pipeline steps.

Each step has a fingerprint made of named components, for example
    en_first = {features: <hash of the feature step>, config.en_first: <hash>, config.final_model: <hash>,
                code: <hash of the step's source files>}
The step's key is the hash of its components. If a result with that key is in the cache it is loaded
instead of recomputed. Downstream steps include upstream keys, so a change anywhere re-runs exactly the
steps that depend on it and nothing else.

Data are hashed by content after loading (not by file name), so renaming a file or a column that the
config maps does not invalidate the cache, while any change to a value does.
"""
import datetime as dt
import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _h(b):
    return hashlib.sha256(b).hexdigest()


def _norm(o):
    """Make equal settings hash equally: 5 == 5.0, tuples == lists, keys sorted by json."""
    if isinstance(o, dict):
        return {str(k): _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, (int, float, np.integer, np.floating)):
        return float(o)
    return str(o)


def hash_obj(obj):
    """Stable hash of config-like objects (dicts, lists, numbers, strings)."""
    return _h(json.dumps(_norm(obj), sort_keys=True).encode())


def hash_frame(df):
    """Content hash of data: values and index. For a DataFrame the column labels count too (they are the
    variable mnemonics); a Series' own name does not, so renaming the balance column changes nothing."""
    if isinstance(df, pd.Series):
        df = df.rename(None).to_frame()
        cols = b""
    else:
        cols = json.dumps([str(c) for c in df.columns]).encode()
    rows = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return _h(rows + cols)


def hash_code(*relpaths):
    """Hash of source files under src/ (so editing the code re-runs the step)."""
    m = hashlib.sha256()
    for rel in sorted(relpaths):
        with open(os.path.join(SRC, rel), "rb") as f:
            m.update(rel.encode() + f.read())
    return m.hexdigest()


class StepCache:
    def __init__(self, cache_dir, enabled=True, force=False, force_steps=()):
        self.dir = cache_dir
        self.enabled = enabled
        self.force = force
        self.force_steps = set(force_steps or ())
        self.records = {}
        os.makedirs(cache_dir, exist_ok=True)

    def run(self, step, components, fn, log=print):
        """Return fn()'s result, from cache when the fingerprint is unchanged."""
        key = hash_obj(components)
        path = os.path.join(self.dir, f"{step.replace(' ', '_').replace(':', '_')}__{key[:16]}.pkl")
        t0 = time.time()
        forced = self.force or step in self.force_steps or step.split(":")[0] in self.force_steps
        if self.enabled and not forced and os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            status = "cached"
        else:
            obj = fn()
            if self.enabled:
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            status = "forced" if forced else ("computed" if self.enabled else "computed (cache off)")
        self.records[step] = {"key": key, "status": status, "seconds": round(time.time() - t0, 2),
                              "components": {k: (v if isinstance(v, str) else hash_obj(v)) for k, v in components.items()}}
        log(f"  [{status:>8s}] {step:<28s} key {key[:10]}  ({self.records[step]['seconds']}s)")
        return obj

    def write_manifest(self, out_dir, log=print):
        """Save this run's fingerprints and report which components changed since the previous run."""
        mpath = os.path.join(out_dir, "run_manifest.json")
        prev = {}
        if os.path.exists(mpath):
            with open(mpath) as f:
                prev = json.load(f).get("steps", {})
        rows = []
        for step, rec in self.records.items():
            old = prev.get(step)
            if old is None:
                changed = "new step"
            elif old["key"] == rec["key"]:
                changed = "unchanged"
            else:
                diff = [k for k, v in rec["components"].items() if old["components"].get(k) != v]
                diff += [k for k in old["components"] if k not in rec["components"]]
                changed = "changed: " + ", ".join(diff)
            rec["vs_previous_run"] = changed
            rows.append({"time": dt.datetime.now().isoformat(timespec="seconds"), "step": step,
                         "status": rec["status"], "key": rec["key"][:16], "seconds": rec["seconds"],
                         "vs_previous_run": changed})
        with open(mpath, "w") as f:
            json.dump({"time": dt.datetime.now().isoformat(timespec="seconds"), "steps": self.records}, f, indent=2)
        hist = os.path.join(out_dir, "run_history.csv")
        pd.DataFrame(rows).to_csv(hist, mode="a", header=not os.path.exists(hist), index=False)
        log("\nStep log (vs previous run):")
        for r in rows:
            log(f"  {r['step']:<28s} {r['status']:>8s}   {r['vs_previous_run']}")
        return pd.DataFrame(rows)
