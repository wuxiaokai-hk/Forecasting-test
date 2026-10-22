"""Write tables (CSV + one Excel workbook per folder), charts and a markdown summary."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "lines.linewidth": 2, "legend.frameon": False,
    "legend.labelcolor": INK2, "figure.dpi": 110,
})


def write_tables(folder, tables):
    os.makedirs(folder, exist_ok=True)
    with pd.ExcelWriter(os.path.join(folder, "tables.xlsx"), engine="openpyxl") as w:
        for name, df in tables.items():
            df.to_csv(os.path.join(folder, f"{name}.csv"), index=False)
            sheet = re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31]
            df.to_excel(w, sheet_name=sheet, index=False)
            ws = w.sheets[sheet]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200]) + 2
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width))


def chart_horizon(path, by_h):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, d) in enumerate(by_h.groupby("model", sort=False)):
        ax.plot(d["horizon_q"], d["MAPE_exCOVID_pct"], color=PAL[i % 8], marker="o", ms=4,
                ls="--" if "benchmark" in name.lower() else "-", label=name)
    ax.set_xlabel("Quarters ahead")
    ax.set_ylabel("MAPE of the balance (%)")
    ax.set_ylim(0, None)
    ax.set_title("Rolling backtest: balance error by horizon (dummy quarters excluded)")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_scenarios(path, scen, level, models):
    fig, axs = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.4), sharey=True, squeeze=False)
    hx = level.index[-16:].to_timestamp()
    cols = [PAL[0], PAL[2], PAL[3], PAL[7], PAL[6], PAL[4]]
    for ax, m in zip(axs[0], models):
        ax.plot(hx, level.iloc[-16:] / 1000, color=INK, label="Actual")
        for i, (sname, d) in enumerate(scen[scen["model"] == m].groupby("scenario", sort=False)):
            xs = pd.PeriodIndex(d["date"], freq="Q").to_timestamp()
            ax.plot(np.r_[hx[-1:], xs], np.r_[level.iloc[-1], d["balance_forecast"].values] / 1000,
                    color=cols[i % len(cols)], label=sname)
        ax.set_title(m)
        ax.legend(fontsize=9)
    axs[0][0].set_ylabel("Balance (thousands of input units)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_stability(path, freq, threshold, top=25):
    d = freq.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(d) + 1.2))
    ax.barh(d["feature"], d["selection_frequency"],
            color=[PAL[0] if v >= threshold else "#b9b8b2" for v in d["selection_frequency"]], height=0.65)
    ax.axvline(threshold, color=INK2, ls=":", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of subsamples in which the feature was selected")
    ax.set_title("Elastic Net stability selection (blue = kept)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_md(path, cfg, results, comparison, diag, overlap):
    L = [f"# Loan forecasting run: {cfg['run_name']}", "",
         f"Config: `{cfg['config_file']}`  ",
         f"Target: `{cfg['data']['bank']['target_column']}` (organic quarterly log change). "
         f"Estimation {cfg['sample']['estimation_start']}–{cfg['sample']['hist_end']}; backtest origins from "
         f"{cfg['sample']['backtest_start']}, 1–{cfg['sample']['horizon']} quarters ahead.", ""]
    for r in results:
        L += [f"## {r['name']}", "", "| Step | Count |", "|---|---|"] + [f"| {a} | {b} |" for a, b in r["funnel"]]
        L += ["", "Final drivers: " + ", ".join(f"`{f}`" for f in r["features"]), ""]
    L += ["## Comparison (average balance error over 1–H quarters, dummy quarters excluded)", "",
          comparison.to_markdown(index=False) if hasattr(comparison, "to_markdown") else comparison.to_string(index=False), ""]
    if overlap is not None:
        L += [f"Drivers chosen by both approaches: {overlap or 'none'}", ""]
    L += ["## Diagnostics", "", diag.to_markdown(index=False) if hasattr(diag, "to_markdown") else diag.to_string(index=False)]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
