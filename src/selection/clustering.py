"""Group features whose |correlation| exceeds a threshold and keep one representative per group."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cluster(F, corr_threshold):
    """Average-linkage clustering on distance 1 - |corr|. Returns a Series of cluster labels."""
    if F.shape[1] == 1:
        return pd.Series([1], index=F.columns)
    C = F.corr().abs().fillna(0).values.copy()
    np.fill_diagonal(C, 1.0)
    Z = linkage(squareform(np.clip(1 - C, 0, None), checks=False), method="average")
    return pd.Series(fcluster(Z, t=1 - corr_threshold, criterion="distance"), index=F.columns)


def representatives(table, score_col, primary_geography=None, tolerance=0.75):
    """Within each cluster keep the highest score, unless a primary-geography member scores at least
    `tolerance` x the best, in which case keep the best primary-geography member."""
    keep = []
    for _, g in table.groupby("cluster"):
        g = g.sort_values(score_col, ascending=False)
        prim = g[g["geography"] == primary_geography] if primary_geography else g.iloc[0:0]
        if len(prim) and prim[score_col].iloc[0] >= tolerance * g[score_col].iloc[0]:
            keep.append(prim.index[0])
        else:
            keep.append(g.index[0])
    out = table.copy()
    out["is_representative"] = out.index.isin(keep)
    out["cluster_size"] = out.groupby("cluster")["feature"].transform("count")
    return out
