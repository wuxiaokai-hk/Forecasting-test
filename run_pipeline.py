"""
Loan balance forecasting pipeline - entry point.

    python run_pipeline.py                                   # uses config/example_config.yaml
    python run_pipeline.py --config config/my_portfolio.yaml
    python run_pipeline.py --config config/my_portfolio.yaml --approach en_first

Everything specific to a data set (file names, sheet names, column names, dates, settings) is in the
config file, so the same code runs on any portfolio by pointing it at another config.

Folders
  config/            run configurations (one YAML per data set)
  data/              input files (simulated example data)
  src/utils/         config loading, package set-up, reading files
  src/features/      transformations, stationarity, candidate features
  src/selection/     univariate screen, clustering, Elastic Net (+ stability selection)
  src/modelling/     ARDL model, shared final search, backtest / diagnostics / scenarios
  src/pipelines/     screen_first.py (approach A), en_first.py (approach B)
  src/reporting/     tables, charts, summary
  tools/             simulate_data.py (makes the example data)
  outputs/<run_name>/screen_first, en_first, comparison
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils.env import ensure_packages  # noqa: E402

ensure_packages()

import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from src.features.transforms import all_features, candidate_forms  # noqa: E402
from src.modelling.ardl import ARDL  # noqa: E402
from src.modelling.evaluate import evaluate  # noqa: E402
from src.pipelines import en_first, screen_first  # noqa: E402
from src.reporting import report  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.cache import StepCache, hash_code, hash_frame, hash_obj  # noqa: E402
from src.utils.io import Context, log  # noqa: E402

warnings.filterwarnings("ignore")
APPROACHES = {"screen_first": screen_first, "en_first": en_first}


def main():
    ap = argparse.ArgumentParser(description="Loan balance forecasting pipeline")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "example_config.yaml"))
    ap.add_argument("--approach", choices=["both", "screen_first", "en_first"], default=None,
                    help="overrides 'approaches' in the config")
    ap.add_argument("--force", action="store_true", help="recompute every step (ignore the cache)")
    ap.add_argument("--force-step", action="append", default=[],
                    help="recompute one step: features, screen_first, en_first or evaluate (repeatable)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.approach is None:
        approaches = list(cfg["approaches"])
    else:
        approaches = ["screen_first", "en_first"] if args.approach == "both" else [args.approach]
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    log(f"Run '{cfg['run_name']}' -> {out}")

    c = cfg["cache"]
    cache = StepCache(c["dir"], enabled=c["enabled"] and not args.no_cache, force=args.force,
                      force_steps=args.force_step)

    log("Loading data ...")
    ctx = Context(cfg)
    log(f"  target: {cfg['data']['bank']['target_column']}, {ctx.idx[0]}–{ctx.idx[-1]} ({len(ctx.idx)} quarters); "
        f"{ctx.base.shape[1]} macro columns; {len(ctx.macro)} scenarios")

    # ---- fingerprints of the inputs (content, not file names) ----
    h_y, h_level, h_meta = hash_frame(ctx.y), hash_frame(ctx.level), hash_frame(ctx.meta)
    h_hist = hash_frame(ctx.hist)
    h_scen = {k: hash_frame(v) for k, v in ctx.macro.items()}
    cs = {k: hash_obj(cfg[k]) for k in ("sample", "features", "screen_first", "en_first", "final_model")}
    log("Steps (cached = loaded because nothing it depends on changed):")
    cache.run("data", {"target": h_y, "balance": h_level, "metadata": h_meta, "macro_history": h_hist,
                       "scenarios": hash_obj(h_scen)}, lambda: True, log)

    def build_features():
        forms_ = candidate_forms(cfg, ctx.hist, ctx.meta)
        if forms_.empty:
            raise RuntimeError("No candidate variables left: check allowed_geographies / excluded_themes / metadata")
        return forms_, all_features(cfg, ctx, forms_)
    feat_comp = {"macro_history": h_hist, "metadata": h_meta, "config.features": cs["features"],
                 "config.sample": cs["sample"], "code": hash_code("features/transforms.py", "utils/io.py")}
    forms, F = cache.run("features", feat_comp, build_features, log)
    h_feat = hash_obj(feat_comp)
    log(f"  {forms['mnemonic'].nunique()} variables, {int(forms['stationary'].sum())} stationary forms, "
        f"{F.shape[1]} features (with lags)")
    report.write_tables(os.path.join(out, "common"), {"stationarity_tests": forms})

    code_files = {
        "screen_first": ["selection/screening.py", "selection/clustering.py", "modelling/search.py",
                         "modelling/ardl.py", "pipelines/screen_first.py"],
        "en_first": ["selection/elastic_net.py", "selection/screening.py", "selection/clustering.py",
                     "modelling/search.py", "modelling/ardl.py", "pipelines/en_first.py"],
    }
    results = []
    for a in approaches:
        comp_a = {"features": h_feat, "target": h_y, "metadata": h_meta, f"config.{a}": cs[a],
                  "config.final_model": cs["final_model"], "config.sample": cs["sample"],
                  "code": hash_code(*code_files[a])}
        r = cache.run(a, comp_a, lambda a=a: APPROACHES[a].run(cfg, ctx, F, forms), log)
        log(f"  {a}: drivers {r['features']}")
        report.write_tables(os.path.join(out, a), r["tables"])
        if "B2_en_stability" in r["tables"]:
            report.chart_stability(os.path.join(out, a, "en_stability.png"), r["tables"]["B2_en_stability"],
                                   cfg["en_first"]["stability"]["threshold"])
        results.append(r)

    log("Evaluating final models ...")
    models = [ARDL("AR(1) benchmark", [])] + [ARDL(r["name"], r["features"]) for r in results]
    ev_code = hash_code("modelling/ardl.py", "modelling/evaluate.py", "modelling/search.py", "features/transforms.py")
    ev = {}
    for m in models:
        comp_m = {"drivers": hash_obj(m.features), "target": h_y, "balance": h_level, "metadata": h_meta,
                  "scenarios": hash_obj(h_scen), "config.sample": cs["sample"],
                  "config.deflator": hash_obj(cfg["features"]["deflator"]), "code": ev_code}
        ev[m.name] = cache.run(f"evaluate:{m.name}", comp_m, lambda m=m: evaluate(ctx, m), log)
    by_h = pd.concat([v["by_horizon"] for v in ev.values()])
    H = cfg["sample"]["horizon"]
    comp = by_h.groupby("model", sort=False).agg(avg_MAPE_exCOVID_pct=("MAPE_exCOVID_pct", "mean"),
                                                 avg_MAPE_all_pct=("MAPE_all_pct", "mean"),
                                                 avg_bias_exCOVID_pct=("bias_exCOVID_pct", "mean")).reset_index()
    for h in sorted({1, min(4, H), H}):
        comp[f"MAPE_h{h}_exCOVID_pct"] = by_h[by_h["horizon_q"] == h].set_index("model").loc[comp["model"], "MAPE_exCOVID_pct"].values
    bench = comp.loc[comp["model"] == "AR(1) benchmark", "avg_MAPE_exCOVID_pct"].iloc[0]
    comp["improvement_vs_benchmark_pct"] = (1 - comp["avg_MAPE_exCOVID_pct"] / bench) * 100
    comp["drivers"] = [", ".join(m.features) or "(none)" for m in models]
    comp = comp.round(3)
    diag = pd.DataFrame([v["diagnostics"] for v in ev.values()])
    drivers = pd.concat([v["drivers"] for v in ev.values()])
    scen = pd.concat([v["scenarios"] for v in ev.values()])
    yearend = scen[scen["date"].str.endswith("Q4")].pivot_table(index=["model", "date"], columns="scenario",
                                                                 values="balance_forecast").round(1).reset_index()
    coef_paths = pd.concat([v["coef_path"].assign(model=k).reset_index().rename(columns={"index": "origin"})
                            for k, v in ev.items()])
    overlap = None
    if len(results) == 2:
        overlap = ", ".join(sorted(set(results[0]["features"]) & set(results[1]["features"])))

    cdir = os.path.join(out, "comparison")
    report.write_tables(cdir, {"1_model_comparison": comp, "2_backtest_by_horizon": by_h.round(4),
                               "3_drivers": drivers, "4_diagnostics": diag, "5_scenario_yearend": yearend,
                               "6_scenario_forecasts": scen, "7_coefficient_paths": coef_paths})
    report.chart_horizon(os.path.join(cdir, "backtest_by_horizon.png"), by_h)
    report.chart_scenarios(os.path.join(cdir, "scenarios.png"), scen, ctx.level, [r["name"] for r in results])
    report.summary_md(os.path.join(out, "summary.md"), cfg, results, comp, diag, overlap)

    log("\nResults (average balance error, 1–%d quarters, dummy quarters excluded):" % H)
    for _, r in comp.iterrows():
        log(f"  {r['model']:<20s} {r['avg_MAPE_exCOVID_pct']:.2f}%   drivers: {r['drivers']}")
    steps = cache.write_manifest(out, log)
    report.write_tables(os.path.join(out, "comparison"), {"8_run_steps": steps})
    log(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
