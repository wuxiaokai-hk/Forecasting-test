"""ARDL model used as the final model by both approaches:

    y_t = a + rho*y_{t-1} + sum_j b_j*x_{j,t-l_j} + sum_q g_q*D_{q,t} + e_t

Estimated by OLS with Newey-West standard errors; forecasts are iterated, with the lagged target set to
its own forecast and the dummies set to zero.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..features.transforms import feature_matrix


def ols_np(X, y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = e @ e / max(len(y) - X.shape[1], 1)
    return b, np.sqrt(np.clip(np.diag(XtX_inv) * s2, 1e-30, None)), e


def vif(F):
    if F.shape[1] < 2:
        return np.ones(F.shape[1])
    try:
        return np.diag(np.linalg.inv(np.corrcoef(F, rowvar=False)))
    except np.linalg.LinAlgError:
        return np.full(F.shape[1], np.inf)


def make_panel(ctx, mdf, features):
    """Quarterly panel for one scenario: target, dummies and features (history + scenario path)."""
    idx = mdf.index[mdf.index >= ctx.idx[0]]
    P = pd.DataFrame(index=idx)
    P["y"] = ctx.y.reindex(idx)
    for c in ctx.dummies.columns:
        P[c] = ctx.dummies[c].reindex(idx).fillna(0.0)
    F = feature_matrix(mdf, features, ctx.cfg["features"]["deflator"]).reindex(idx)
    return pd.concat([P, F], axis=1)


class ARDL:
    def __init__(self, name, features, ar=True):
        self.name, self.features, self.ar = name, list(features), ar

    def _X(self, P):
        X = pd.DataFrame({"const": 1.0}, index=P.index)
        if self.ar:
            X["y_lag1"] = P["y"].shift(1)
        for f in self.features:
            X[f] = P[f]
        for c in [c for c in P.columns if c.startswith("D_")]:
            X[c] = P[c]
        return X

    def fit(self, P, start, end):
        X = self._X(P).loc[start:end]
        yv = P["y"].loc[start:end]
        X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
        ok = X.notna().all(axis=1) & yv.notna()
        self.res = sm.OLS(yv[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        self.params = self.res.params
        return self

    def forecast(self, P, origin, H):
        pos = P.index.get_loc(origin)
        H = min(H, len(P) - pos - 1)
        yprev = P["y"].iloc[pos]
        out = np.zeros(H)
        for h in range(1, H + 1):
            row = P.iloc[pos + h]
            v = self.params.get("const", 0.0) + self.params.get("y_lag1", 0.0) * yprev
            v += sum(self.params[f] * row[f] for f in self.features if f in self.params)
            out[h - 1] = v
            yprev = v
        return out
