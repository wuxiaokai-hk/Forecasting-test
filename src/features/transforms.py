"""Variable transformations, stationarity tests and the candidate feature list.

A feature is one variable in one transformation at one lag, named  'MNEMONIC|transform|L<lag>'.
All rolling windows are one-sided (past data only).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

TR_DESC = {
    "dlog": "q/q % growth", "yoy": "y/y % growth", "dlog2": "2-quarter % growth",
    "acc": "acceleration (change in y/y growth vs a year ago)", "gap": "% gap from 3-year trailing average",
    "rdlog": "real q/q % growth", "ryoy": "real y/y % growth",
    "diff": "q/q change", "diff2": "2-quarter change", "diff4": "y/y change", "lvl": "level",
    "rgap": "gap from 3-year trailing average (pp)",
}
RATE_TRS = {"diff", "diff2", "diff4", "lvl", "rgap"}


def transform(mdf, code, tr, deflator=None):
    x = mdf[code]
    if tr == "diff":
        return x.diff()
    if tr == "diff2":
        return x.diff(2)
    if tr == "diff4":
        return x.diff(4)
    if tr == "lvl":
        return x
    if tr == "rgap":
        return x - x.rolling(12, min_periods=12).mean()
    lx = np.log(x)
    if tr == "dlog":
        return lx.diff()
    if tr == "yoy":
        return lx.diff(4)
    if tr == "dlog2":
        return lx.diff(2)
    if tr == "acc":
        return lx.diff(4) - lx.diff(4).shift(4)
    if tr == "gap":
        return lx - lx.rolling(12, min_periods=12).mean()
    if tr in ("rdlog", "ryoy"):
        k = 1 if tr == "rdlog" else 4
        return lx.diff(k) - np.log(mdf[deflator]).diff(k)
    raise ValueError(f"unknown transform {tr}")


def feature_name(code, tr, lag):
    return f"{code}|{tr}|L{lag}"


def parse_feature(name):
    code, tr, lag = name.rsplit("|", 2)
    return code, tr, int(lag[1:])


def feature_series(mdf, name, deflator=None):
    code, tr, lag = parse_feature(name)
    return transform(mdf, code, tr, deflator).shift(lag)


def feature_matrix(mdf, names, deflator=None):
    return pd.DataFrame({n: feature_series(mdf, n, deflator) for n in names}, index=mdf.index)


def describe(name, meta):
    code, tr, lag = parse_feature(name)
    d = meta.loc[code, "description"] if code in meta.index else code
    return f"{d} ({TR_DESC.get(tr, tr)}, {'same quarter' if lag == 0 else f'lag {lag}Q'})"


def candidate_forms(cfg, hist, meta):
    """Relevance filter + every transformation of every variable + ADF test. One row per (variable, transform)."""
    f = cfg["features"]
    rows = []
    for code in hist.columns:
        if code not in meta.index:
            continue
        geo, theme = meta.loc[code, "geography"], meta.loc[code, "theme"]
        if f["allowed_geographies"] and not any(geo.startswith(g) for g in f["allowed_geographies"]):
            continue
        if theme in (f["excluded_themes"] or []):
            continue
        if f["require_sign_prior"] and meta.loc[code, "expected_sign"] == 0:
            continue
        x = hist[code].dropna()
        is_level = meta.loc[code, "transform"] == "dlog" and len(x) and (x > 0).all()
        trs = list(f["transforms_level"]) if is_level else list(f["transforms_rate"])
        if is_level and theme in (f["real_themes"] or []) and f["deflator"] in hist.columns:
            trs += ["rdlog", "ryoy"]
        for tr in trs:
            s = transform(hist, code, tr, f["deflator"]).dropna()
            try:
                p = adfuller(s, autolag="AIC")[1] if (len(s) >= 30 and s.std() > 0) else 1.0
            except Exception:
                p = 1.0
            rows.append({"mnemonic": code, "transform": tr, "adf_p": p, "stationary": p < f["adf_p"],
                         "theme": theme, "geography": geo, "expected_sign": meta.loc[code, "expected_sign"]})
    return pd.DataFrame(rows)


def all_features(cfg, ctx, forms):
    """All stationary (variable, transform) pairs at every lag, with no gaps in the estimation window."""
    names = [feature_name(r.mnemonic, r.transform, l) for r in forms[forms["stationary"]].itertuples()
             for l in range(cfg["features"]["max_lag"] + 1)]
    F = feature_matrix(ctx.hist, names, cfg["features"]["deflator"])
    window = F.iloc[ctx.est0:ctx.end + 1]
    keep = window.columns[window.notna().all() & (window.std() > 0)]
    return F[keep]
