"""Read the bank, event, macro and metadata files using the names given in the config."""
import os

import numpy as np
import pandas as pd


def log(msg):
    print(msg, flush=True)


def read_table(path, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    return pd.read_csv(path)


def to_quarters(values):
    """'2020Q1', '2020 Q1', '2020-03-31', datetime ... -> quarterly PeriodIndex."""
    s = pd.Series(values).astype(str).str.strip()
    if s.str.match(r"^\d{4}\s*[Qq]\d$").all():
        return pd.PeriodIndex(s.str.replace(" ", "").str.upper(), freq="Q")
    return pd.PeriodIndex(pd.to_datetime(s), freq="Q")


def _need(df, cols, label):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{label}: column(s) {miss} not found. Available: {list(df.columns)[:15]} ...")


def load_bank(cfg):
    b = cfg["data"]["bank"]
    df = read_table(b["file"], b["sheet"])
    _need(df, [b["date_column"], b["target_column"]], "bank file")
    s = pd.to_numeric(df[b["target_column"]], errors="coerce")
    s.index = to_quarters(df[b["date_column"]])
    s = s.groupby(level=0).last() if b["frequency"].upper() == "M" else s[~s.index.duplicated(keep="last")]
    return s.sort_index().loc[:pd.Period(cfg["sample"]["hist_end"], freq="Q")].dropna()


def organic_growth(cfg, level):
    """y_t = ln((L_t - A_t) / L_{t-1}), A_t = inorganic events in quarter t."""
    y = np.log(level).diff()
    e = cfg["data"]["events"]
    if e["file"] and os.path.exists(e["file"]):
        ev = read_table(e["file"], e["sheet"])
        _need(ev, [e["date_column"], e["series_column"], e["amount_column"]], "events file")
        ev = ev[ev[e["series_column"]].astype(str) == cfg["data"]["bank"]["target_column"]]
        for q, a in zip(to_quarters(ev[e["date_column"]]), ev[e["amount_column"]]):
            if q in level.index and q != level.index[0]:
                y.loc[q] = np.log((level.loc[q] - float(a)) / level.shift(1).loc[q])
    return y


def load_macro(cfg):
    m = cfg["data"]["macro"]
    out = {}
    for name, f in m["scenarios"].items():
        df = read_table(f, m["sheet"])
        _need(df, [m["date_column"]], f"macro file '{name}'")
        idx = to_quarters(df[m["date_column"]])
        df = df.drop(columns=m["date_column"]).apply(pd.to_numeric, errors="coerce")
        df.index = idx
        out[name] = df.sort_index()
    return out


def load_metadata(cfg):
    md = cfg["data"]["metadata"]
    df = read_table(md["file"], md["sheet"])
    cmap = md["columns"]
    _need(df, list(cmap.values()), "metadata file")
    df = df.rename(columns={v: k for k, v in cmap.items()})[list(cmap.keys())]
    df["mnemonic"] = df["mnemonic"].astype(str)
    df["expected_sign"] = pd.to_numeric(df["expected_sign"], errors="coerce").fillna(0).astype(int)
    df["transform"] = df["transform"].astype(str).str.lower().str.strip()
    for c in ("description", "geography", "theme"):
        df[c] = df[c].fillna("").astype(str)
    return df.set_index("mnemonic")


def add_constructed(cfg, macro, meta):
    """Build the variables listed under data.constructed in every scenario."""
    for c in cfg["data"].get("constructed") or []:
        try:
            for name in macro:
                env = {"log": np.log, "exp": np.exp}
                macro[name][c["name"]] = macro[name].eval(c["expression"], local_dict=env, engine="python")
        except Exception as e:
            log(f"  constructed variable {c.get('name')} skipped: {e}")
            continue
        meta.loc[c["name"]] = {"description": c.get("description", c["name"]), "geography": c.get("geography", ""),
                               "theme": c.get("theme", "Constructed"), "transform": c.get("transform", "diff"),
                               "expected_sign": int(c.get("expected_sign", 0))}
    return macro, meta


class Context:
    """Everything both approaches need: target, macro data, metadata, sample indices."""

    def __init__(self, cfg):
        self.cfg = cfg
        level = load_bank(cfg)
        self.level = level
        self.y = organic_growth(cfg, level)
        self.logL = np.log(level.iloc[0]) + self.y.fillna(0).cumsum()      # organic log level
        self.macro = load_macro(cfg)
        self.meta = load_metadata(cfg)
        self.macro, self.meta = add_constructed(cfg, self.macro, self.meta)
        self.base = self.macro[cfg["data"]["macro"]["base_scenario"]]
        unknown = [c for c in self.base.columns if c not in self.meta.index]
        if unknown:
            log(f"  {len(unknown)} macro columns have no metadata row and are ignored (e.g. {unknown[:3]})")
        hist_end = pd.Period(cfg["sample"]["hist_end"], freq="Q")
        self.idx = self.y.index                      # history quarters
        self.hist = self.base.reindex(self.idx)
        self.est0 = int(np.where(self.idx == pd.Period(cfg["sample"]["estimation_start"], freq="Q"))[0][0])
        self.end = int(np.where(self.idx == hist_end)[0][0]) if hist_end in self.idx else len(self.idx) - 1
        self.dummies = pd.DataFrame({f"D_{q}": (self.idx == pd.Period(q, freq="Q")).astype(float)
                                     for q in cfg["sample"]["dummy_quarters"]}, index=self.idx)
