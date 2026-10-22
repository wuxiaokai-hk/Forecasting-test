"""Univariate screen: each candidate is added on its own to a base regression with lagged loan growth
and the dummies, using Newey-West (HAC) standard errors.

    y_t = a + rho*y_{t-1} + sum_q g_q*D_{q,t} + beta*z_{t-l} + e_t
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..features.transforms import parse_feature


def controls(ctx):
    C = pd.DataFrame({"const": 1.0, "y_lag1": ctx.y.shift(1)}, index=ctx.idx)
    return pd.concat([C, ctx.dummies], axis=1)


def screen(cfg, ctx, F):
    rows_ = slice(ctx.est0, ctx.end + 1)
    yy = ctx.y.iloc[rows_]
    C = controls(ctx).iloc[rows_]
    C = C.loc[:, (C.std() > 0) | (C.columns == "const")]
    out = []
    for name in F.columns:
        x = F[name].iloc[rows_]
        res = sm.OLS(yy, C.assign(x=x)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        code, tr, lag = parse_feature(name)
        out.append({"feature": name, "mnemonic": code, "transform": tr, "lag": lag, "coef": res.params["x"],
                    "t_hac": res.tvalues["x"], "p_hac": res.pvalues["x"]})
    sc = pd.DataFrame(out)
    m = ctx.meta
    sc["theme"] = sc["mnemonic"].map(m["theme"])
    sc["geography"] = sc["mnemonic"].map(m["geography"])
    sc["description"] = sc["mnemonic"].map(m["description"])
    sc["expected_sign"] = sc["mnemonic"].map(m["expected_sign"]).astype(int)
    sc["sign_ok"] = (sc["expected_sign"] == 0) | (np.sign(sc["coef"]) == sc["expected_sign"])
    sc["abs_t"] = sc["t_hac"].abs()
    sc["passed"] = sc["sign_ok"] & (sc["p_hac"] < cfg["screen_first"]["screen_p"])
    best = sc.sort_values(["passed", "abs_t"], ascending=[False, False]).groupby("mnemonic").head(1)
    return sc.sort_values("abs_t", ascending=False), best.sort_values(["passed", "abs_t"], ascending=[False, False])
