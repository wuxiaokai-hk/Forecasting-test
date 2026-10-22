"""Shared final step: try every combination of min_vars..max_vars candidates, keep those that satisfy the
economic and statistical rules, rank them on out-of-sample balance error, and pick the simplest model
within `selection_tolerance` of the best."""
import itertools

import numpy as np
import pandas as pd

from .ardl import ols_np, vif


def backtest_origins(ctx):
    start = pd.Period(ctx.cfg["sample"]["backtest_start"], freq="Q")
    return [i for i, p in enumerate(ctx.idx) if p >= start and i < ctx.end]


def covid_mask(ctx, origins, H):
    dq = [pd.Period(q, freq="Q") for q in ctx.cfg["sample"]["dummy_quarters"]]
    M = np.zeros((len(origins), H), dtype=bool)
    if not dq:
        return M
    a, b = min(dq), max(dq)
    for i, o in enumerate(origins):
        for h in range(1, H + 1):
            if o + h < len(ctx.idx):
                win = ctx.idx[o + 1:o + h + 1]
                M[i, h - 1] = ((win >= a) & (win <= b)).any()
    return M


def fast_backtest(F, y, D, est0, origins, H):
    """Level % errors of iterated ARDL forecasts. F: features (T x k), y: target, D: dummies (T x d)."""
    T = len(y)
    E = np.full((len(origins), H), np.nan)
    for i, o in enumerate(origins):
        rows = np.arange(est0, o + 1)
        Dk = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
        X = np.column_stack([np.ones(len(rows)), y[rows - 1], Dk, F[rows]])
        b = np.linalg.lstsq(X, y[rows], rcond=None)[0]
        c, rho, beta = b[0], b[1], b[2 + Dk.shape[1]:]
        yprev, cp, ca = y[o], 0.0, 0.0
        for h in range(1, H + 1):
            t = o + h
            if t >= T:
                break
            yh = c + rho * yprev + F[t] @ beta
            cp += yh
            ca += y[t]
            E[i, h - 1] = np.exp(cp - ca) - 1
            yprev = yh
    return E


def exhaustive_search(cfg, ctx, F, info, label):
    """F: candidate features on the history index; info: DataFrame indexed by feature with theme, expected_sign."""
    fm = cfg["final_model"]
    feats = list(F.columns)
    themes = info.loc[feats, "theme"].values
    signs = info.loc[feats, "expected_sign"].values
    y = ctx.y.values
    D = ctx.dummies.values
    rows = np.arange(ctx.est0, ctx.end + 1)
    Fv = F.values
    Dr = D[rows][:, D[rows].std(axis=0) > 0] if D.shape[1] else np.zeros((len(rows), 0))
    base = np.column_stack([np.ones(len(rows)), y[rows - 1], Dr])
    n = len(rows)
    res = []
    for k in range(fm["min_vars"], fm["max_vars"] + 1):
        for combo in itertools.combinations(range(len(feats)), k):
            th = themes[list(combo)]
            if fm["one_per_theme"] and len(set(th)) < k:
                continue
            if fm["required_themes"] and not set(fm["required_themes"]) <= set(th):
                continue
            Fk = Fv[rows][:, combo]
            X = np.column_stack([base, Fk])
            b, se, e = ols_np(X, y[rows])
            bk, tk = b[-k:], b[-k:] / se[-k:]
            if np.any(np.abs(tk) < fm["t_min"]):
                continue
            sg = signs[list(combo)]
            if np.any((sg != 0) & (np.sign(bk) != sg)):
                continue
            v = vif(Fk)
            if np.max(v) > fm["vif_max"]:
                continue
            ssr = e @ e
            res.append({"combo": combo, "k": k, "bic": n * np.log(ssr / n) + X.shape[1] * np.log(n),
                        "min_abs_t": float(np.min(np.abs(tk))), "max_vif": float(np.max(v))})
    if not res:
        raise RuntimeError(f"[{label}] no combination passed the rules: relax t_min / vif_max / one_per_theme "
                           f"or give the search more candidates.")
    R = pd.DataFrame(res).sort_values("bic").head(fm["n_oos_candidates"]).reset_index(drop=True)
    H = cfg["sample"]["horizon"]
    origins = backtest_origins(ctx)
    M = covid_mask(ctx, origins, H)
    err = []
    for c in R["combo"]:
        E = fast_backtest(Fv[:, list(c)], y, D, ctx.est0, origins, H)
        err.append(np.nanmean(np.abs(np.where(M, np.nan, E))) * 100)
    R["oos_mape_pct"] = err
    R["features"] = R["combo"].apply(lambda c: [feats[i] for i in c])
    R = R.sort_values("oos_mape_pct").reset_index(drop=True)
    best = R["oos_mape_pct"].iloc[0]
    pick = R[R["oos_mape_pct"] <= best * (1 + fm["selection_tolerance"])].sort_values(
        ["k", "min_abs_t"], ascending=[True, False]).index[0]
    R["selected"] = R.index == pick
    R = pd.concat([R.loc[[pick]], R.drop(index=pick)]).reset_index(drop=True)
    R["features"] = R["features"].apply(", ".join)
    return R.drop(columns="combo"), R.loc[0, "features"].split(", ")
