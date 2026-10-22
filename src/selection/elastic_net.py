"""Elastic Net selection (the 'EN first' approach), matching glmnet's conventions.

- Controls (constant, lagged loan growth, dummies) are left unpenalized by partialling them out first
  (equivalent to glmnet penalty.factor = 0 for those columns).
- X and y are standardized with the population standard deviation, as glmnet does.
- Optional sign constraints: columns with an expected sign of -1 are flipped and all coefficients are
  forced >= 0 (glmnet lower.limits); variables with no sign prior enter as a +/- pair.
- lambda is not chosen by cross-validation: it is the first point on the path where `target_n`
  features are selected, so the result is deterministic for given data.
- Stability selection repeats this on subsamples made of consecutive-quarter blocks and reports how
  often each feature is selected (Meinshausen & Buhlmann, 2010).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from ..features.transforms import parse_feature
from .screening import controls


def _prepare(cfg, ctx, F, rows):
    e = cfg["en_first"]
    y = ctx.y.values[rows]
    X = F.values[rows]
    names = list(F.columns)
    C = controls(ctx).values[rows]
    C = C[:, (C.std(axis=0) > 0) | (np.arange(C.shape[1]) == 0)]
    if e["unpenalized_controls"]:
        beta = np.linalg.lstsq(C, np.column_stack([y, X]), rcond=None)[0]
        R = np.column_stack([y, X]) - C @ beta
        y, X = R[:, 0], R[:, 1:]
    else:                                               # controls compete with the macro features
        X = np.column_stack([X, C[:, 1:]])
        names = names + [f"__control{i}" for i in range(C.shape[1] - 1)]
    sd = X.std(axis=0)
    ok = sd > 1e-12
    X, names = (X[:, ok] - X[:, ok].mean(0)) / sd[ok], [n for n, k in zip(names, ok) if k]
    y = (y - y.mean()) / y.std()
    signs = np.ones(len(names))
    if e["sign_constraints"]:
        signs = np.array([ctx.meta.loc[parse_feature(n)[0], "expected_sign"] if not n.startswith("__") else 0
                          for n in names], dtype=float)
        two = signs == 0                                 # no prior: allow both directions
        X = np.column_stack([X * np.where(two, 1, signs), -X[:, two]])
        names = names + [n for n, t in zip(names, two) if t]
        signs = np.r_[np.where(two, 1, signs), -np.ones(two.sum())]
    return X, y, names, signs


def _select_at_target(cfg, X, y, names, signs):
    """Walk down a log-spaced lambda grid with warm starts and stop at the first lambda where
    target_n features are non-zero (same answer as reading it off the full path, much faster)."""
    e = cfg["en_first"]
    n = len(y)
    lam_max = np.max(np.abs(X.T @ y)) / (n * max(e["l1_ratio"], 1e-3))
    grid = lam_max * np.logspace(0, -3, 150)
    is_ctrl = np.array([nm.startswith("__") for nm in names])
    m = ElasticNet(alpha=grid[0], l1_ratio=e["l1_ratio"], positive=e["sign_constraints"], fit_intercept=False,
                   warm_start=True, max_iter=20000, tol=1e-7, selection="cyclic")
    for lam in grid:
        m.set_params(alpha=lam)
        m.fit(X, y)
        nz = (m.coef_ != 0) & ~is_ctrl
        if len(set(np.array(names)[nz])) >= e["target_n"]:
            break
    b = pd.Series(m.coef_ * signs, index=names)
    b = b[b != 0].groupby(level=0).sum()
    b = b[~b.index.str.startswith("__")]
    b = b.reindex(b.abs().sort_values(ascending=False).index).head(e["target_n"])
    return b, lam


def elastic_net_select(cfg, ctx, F):
    rows = np.arange(ctx.est0, ctx.end + 1)
    X, y, names, signs = _prepare(cfg, ctx, F, rows)
    b, lam = _select_at_target(cfg, X, y, names, signs)
    table = pd.DataFrame({"feature": b.index, "std_coef": b.values})
    table["lambda"] = lam
    return table


def stability_selection(cfg, ctx, F):
    st = cfg["en_first"]["stability"]
    rng = np.random.default_rng(st["seed"])
    rows_all = np.arange(ctx.est0, ctx.end + 1)
    blocks = [rows_all[i:i + st["block_length"]] for i in range(0, len(rows_all), st["block_length"])]
    n_take = max(2, int(round(st["fraction"] * len(blocks))))
    counts = pd.Series(0.0, index=F.columns)
    for _ in range(st["n_subsamples"]):
        pick = np.sort(rng.choice(len(blocks), n_take, replace=False))
        rows = np.concatenate([blocks[i] for i in pick])
        X, y, names, signs = _prepare(cfg, ctx, F, rows)
        b, _ = _select_at_target(cfg, X, y, names, signs)
        counts[b.index] += 1
    freq = (counts / st["n_subsamples"]).sort_values(ascending=False)
    return freq[freq > 0].rename("selection_frequency").reset_index().rename(columns={"index": "feature"})
