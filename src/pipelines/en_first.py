"""Approach B - 'Elastic Net first' (the user's approach):
all stationary features -> Elastic Net (lambda set so target_n features are chosen; optional stability
selection) -> group correlated picks, keep one per group -> shared final search."""
import numpy as np
import pandas as pd

from ..features.transforms import parse_feature
from ..modelling.search import exhaustive_search
from ..selection.clustering import cluster, representatives
from ..selection.elastic_net import elastic_net_select, stability_selection
from ..utils.io import log


def run(cfg, ctx, F, forms):
    e = cfg["en_first"]
    log(f"[en_first] Elastic Net on {F.shape[1]} features (target {e['target_n']}) ...")
    full = elastic_net_select(cfg, ctx, F)
    log(f"[en_first] full-sample selection: {list(full['feature'])}")
    tables = {"B1_en_full_sample": full}
    if e["stability"]["enabled"]:
        log(f"[en_first] stability selection on {e['stability']['n_subsamples']} subsamples ...")
        freq = stability_selection(cfg, ctx, F)
        tables["B2_en_stability"] = freq
        sel = freq[freq["selection_frequency"] >= e["stability"]["threshold"]].copy()
        if len(sel) < 2:
            log("[en_first] fewer than 2 features above the stability threshold - using the most frequent ones")
            sel = freq.head(max(3, e["target_n"] // 2)).copy()
        sel["score"] = sel["selection_frequency"]
    else:
        sel = full.copy()
        sel["score"] = sel["std_coef"].abs()
    sel["mnemonic"] = sel["feature"].map(lambda f: parse_feature(f)[0])
    sel["theme"] = sel["mnemonic"].map(ctx.meta["theme"])
    sel["geography"] = sel["mnemonic"].map(ctx.meta["geography"])
    sel["expected_sign"] = sel["mnemonic"].map(ctx.meta["expected_sign"]).astype(int)
    sel["description"] = sel["mnemonic"].map(ctx.meta["description"])
    Fs = F[sel["feature"]].iloc[ctx.est0:ctx.end + 1]
    sel["cluster"] = cluster(Fs, e["group_corr"]).reindex(sel["feature"]).values
    grp = representatives(sel.set_index("feature", drop=False), "score", e["primary_geography"], e["rep_tolerance"])
    cands = grp[grp["is_representative"]].sort_values("score", ascending=False)
    tables["B3_groups"] = grp.reset_index(drop=True)
    log(f"[en_first] {len(sel)} selected -> {len(cands)} groups; searching ...")
    R, chosen = exhaustive_search(cfg, ctx, F[cands["feature"]], cands.reset_index(drop=True).set_index("feature"), "en_first")
    tables["B4_search"] = R
    log(f"[en_first] chosen: {chosen}")
    funnel = [("Candidate variables", forms["mnemonic"].nunique()),
              ("Features into the Elastic Net", F.shape[1]),
              ("Selected by Elastic Net" + (" (stable)" if e["stability"]["enabled"] else ""), len(sel)),
              ("Groups after correlation check", len(cands)), ("Final drivers", len(chosen))]
    return {"name": "Elastic Net first", "features": chosen, "tables": tables, "funnel": funnel}
