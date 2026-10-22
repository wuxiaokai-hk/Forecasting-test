"""Shared evaluation of a final ARDL model: rolling backtest, diagnostics, driver table, scenario forecasts."""
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from ..features.transforms import TR_DESC, RATE_TRS, describe, parse_feature
from .ardl import make_panel, vif
from .search import backtest_origins, covid_mask


def backtest(ctx, model, P):
    H = ctx.cfg["sample"]["horizon"]
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    origins = backtest_origins(ctx)
    E = np.full((len(origins), H), np.nan)
    coefs = []
    for i, o in enumerate(origins):
        lab = ctx.idx[o]
        model.fit(P, start, lab)
        fc = model.forecast(P.loc[:ctx.idx[ctx.end]], lab, H)
        act = ctx.y.values[o + 1:o + 1 + len(fc)]
        E[i, :len(fc)] = np.exp(np.cumsum(fc) - np.cumsum(act)) - 1
        coefs.append(model.params.rename(str(lab)))
    M = covid_mask(ctx, origins, H)
    Ex = np.where(M, np.nan, E)
    by_h = pd.DataFrame({"model": model.name, "horizon_q": np.arange(1, H + 1),
                         "MAPE_all_pct": np.nanmean(np.abs(E), axis=0) * 100,
                         "MAPE_exCOVID_pct": np.nanmean(np.abs(Ex), axis=0) * 100,
                         "bias_exCOVID_pct": np.nanmean(Ex, axis=0) * 100})
    return by_h, pd.DataFrame(coefs)


def diagnostics(ctx, model, P):
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    model.fit(P, start, ctx.idx[ctx.end])
    r = model.res
    e = r.resid
    lb = acorr_ljungbox(e, lags=[4, 8], return_df=True)["lb_pvalue"].values
    try:
        cus = breaks_cusumolsresid(e.values, ddof=r.model.exog.shape[1])[1]
    except Exception:
        cus = np.nan
    exog = pd.DataFrame(r.model.exog, columns=r.model.exog_names)
    feats = [f for f in model.features if f in exog]
    wrong = [f for f in feats if ctx.meta.loc[parse_feature(f)[0], "expected_sign"] not in (0, np.sign(r.params[f]))]
    return {"model": model.name, "n_obs": int(r.nobs), "adj_R2": round(r.rsquared_adj, 3), "BIC": round(r.bic, 1),
            "max p-value of drivers": round(max([r.pvalues[f] for f in feats] or [np.nan]), 4),
            "max VIF": round(float(np.max(vif(exog[feats].values))) if len(feats) > 1 else 1.0, 2),
            "Ljung-Box p (4)": round(lb[0], 3), "Ljung-Box p (8)": round(lb[1], 3),
            "Breusch-Pagan p": round(het_breuschpagan(e, r.model.exog)[1], 3),
            "Jarque-Bera p": round(jarque_bera(e)[1], 3), "Durbin-Watson": round(durbin_watson(e), 2),
            "CUSUM p": round(cus, 3), "wrong-sign drivers": ", ".join(wrong) or "none"}


def driver_table(ctx, model):
    r = model.res
    rho = r.params.get("y_lag1", 0.0)
    rows = []
    for c in r.params.index:
        if c == "const" or c.startswith("D_"):
            continue
        b = r.params[c]
        row = {"model": model.name, "term": c, "coef": b, "se_hac": r.bse[c], "t": r.tvalues[c], "p": r.pvalues[c]}
        if c == "y_lag1":
            row.update(description="Loan growth last quarter (momentum)",
                       reading=f"{b:.2f} of last quarter's growth carries into this quarter")
        else:
            code, tr, lag = parse_feature(c)
            per_unit = b * (1 if tr in RATE_TRS else 0.01) * 100
            unit = "1 pp" if tr in RATE_TRS else "1%"
            when = "in the same quarter" if lag == 0 else f"{lag} quarter{'s' if lag > 1 else ''} later"
            row.update(description=describe(c, ctx.meta),
                       reading=(f"A {unit} rise in {ctx.meta.loc[code, 'description']} [{TR_DESC.get(tr, tr)}] "
                                f"{'raises' if b > 0 else 'lowers'} q/q loan growth by {abs(per_unit):.3f} pp {when}; "
                                f"long-run effect {abs(per_unit) / (1 - rho):.3f} pp"))
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_forecasts(ctx, model):
    start = pd.Period(ctx.cfg["sample"]["estimation_start"], freq="Q")
    last = ctx.level.iloc[-1]
    out = []
    for name, mdf in ctx.macro.items():
        P = make_panel(ctx, mdf, model.features)
        model.fit(P, start, ctx.idx[ctx.end])
        fc = model.forecast(P, ctx.idx[ctx.end], len(P))
        dates = P.index[P.index.get_loc(ctx.idx[ctx.end]) + 1:][:len(fc)]
        lvl = last * np.exp(np.cumsum(fc))
        out += [{"model": model.name, "scenario": name, "date": str(d), "dlog_forecast": f, "balance_forecast": l}
                for d, f, l in zip(dates, fc, lvl)]
    return pd.DataFrame(out)


def evaluate(ctx, model):
    P = make_panel(ctx, ctx.base, model.features)
    by_h, coef_path = backtest(ctx, model, P)
    diag = diagnostics(ctx, model, P)
    drivers = driver_table(ctx, model)
    scen = scenario_forecasts(ctx, model)
    return {"by_horizon": by_h, "coef_path": coef_path, "diagnostics": diag, "drivers": drivers, "scenarios": scen}
