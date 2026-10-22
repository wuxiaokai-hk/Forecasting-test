"""Approach A - 'screen first':
candidates -> univariate screen (significant, expected sign) -> best form/lag per variable
-> cluster near-duplicates, keep one per cluster -> shortlist -> shared final search."""
import pandas as pd

from ..modelling.search import exhaustive_search
from ..selection.clustering import cluster, representatives
from ..selection.screening import screen
from ..utils.io import log


def run(cfg, ctx, F, forms):
    s = cfg["screen_first"]
    log("[screen_first] univariate screen ...")
    sc, best = screen(cfg, ctx, F)
    passed = best[best["passed"]].copy()
    log(f"[screen_first] {len(sc)} features tested, {len(passed)} variables pass")
    if passed.empty:
        raise RuntimeError("[screen_first] no variable passed the screen")
    Fp = F[passed["feature"]].iloc[ctx.est0:ctx.end + 1]
    passed["cluster"] = cluster(Fp, s["cluster_corr"]).reindex(passed["feature"]).values
    clus = representatives(passed.set_index("feature", drop=False), "abs_t", s["primary_geography"], s["rep_tolerance"])
    shortlist = clus[clus["is_representative"]].sort_values("abs_t", ascending=False).head(s["shortlist_max"])
    for th in cfg["final_model"]["required_themes"]:          # make sure a required theme has a candidate
        if th not in set(shortlist["theme"]):
            extra = passed[passed["theme"] == th].head(1)
            if extra.empty:
                raise RuntimeError(f"[screen_first] no passing candidate for required theme '{th}'")
            shortlist = pd.concat([shortlist, extra.set_index("feature", drop=False)])
    log(f"[screen_first] {clus['cluster'].nunique()} clusters -> shortlist of {len(shortlist)}; searching ...")
    R, chosen = exhaustive_search(cfg, ctx, F[shortlist["feature"]], shortlist.reset_index(drop=True).set_index("feature"), "screen_first")
    log(f"[screen_first] chosen: {chosen}")
    funnel = [("Candidate variables", forms["mnemonic"].nunique()),
              ("With a stationary form", forms[forms["stationary"]]["mnemonic"].nunique()),
              ("Features tested (form x lag)", len(sc)), ("Variables passing the screen", len(passed)),
              ("Clusters", int(clus["cluster"].nunique())), ("Shortlist", len(shortlist)), ("Final drivers", len(chosen))]
    tables = {"A1_screen_all": sc, "A2_screen_best_per_variable": best, "A3_clusters": clus.reset_index(drop=True),
              "A4_shortlist": shortlist.reset_index(drop=True), "A5_search": R}
    return {"name": "Screen first", "features": chosen, "tables": tables, "funnel": funnel}
