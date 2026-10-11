"""
Step 6 - Figures for the README.

    pip install matplotlib
    python scripts/06_figures.py

Input : data/stability_selection.csv, data/oos_signal_ic.csv, data/beta_check.csv,
        data/backtest_summary.csv (from step 7; fig4 is skipped if it is missing)
Output: charts/signal_stability/fig1_selection_probability.png
        charts/signal_stability/fig2_train_vs_test_ic.png
        charts/signal_stability/fig3_beta_hedge.png
        charts/signal_stability/fig4_sharpe_intervals.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data")
OUT = Path("charts") / "signal_stability"
PI_THR = 0.80

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

# signals that are all proxies for "how volatile / market-sensitive is this stock"
VOL_BETA = {"beta_126", "vol_21", "vol_63", "parkinson_21", "downside_vol_63",
            "idio_vol_63", "max_ret_21"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def titled(fig, title: str, subtitle: str) -> None:
    h = fig.get_figheight()                       # place in inches so spacing is the same on every figure
    fig.text(0.02, 1 - 0.18 / h, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.50 / h, subtitle, fontsize=9.5, color=INK2, va="top")


def fig1_selection_probability() -> None:
    d = pd.read_csv(DATA / "stability_selection.csv", index_col="signal")["stab_prob"].sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    ax.barh(d.index, d.values, height=0.62, color=BLUE)
    ax.axvline(PI_THR, color=INK, linewidth=1)
    ax.text(PI_THR - 0.012, 0.4, f"selection threshold {PI_THR:.1f}", ha="right", va="center",
            fontsize=9, color=INK2)
    top = d.index[-1]
    ax.text(d[top] + 0.012, len(d) - 1, f"{d[top]:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Selection probability over 200 half-samples of dates")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    titled(fig, "No signal is selected stably",
           "Stability selection on 30 candidate signals, train period 2017-2022")
    fig.subplots_adjust(left=0.2, right=0.96, top=0.895, bottom=0.07)
    fig.savefig(OUT / "fig1_selection_probability.png", dpi=200)
    plt.close(fig)


def fig2_train_vs_test_ic() -> None:
    d = pd.read_csv(DATA / "oos_signal_ic.csv", index_col=0)
    is_vb = d.index.isin(VOL_BETA)
    lim = max(d["train_ic"].abs().max(), d["test_ic"].abs().max()) * 1.18

    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1.2, zorder=1)
    ax.text(lim * 0.97, lim * 0.9, "test = train", ha="right", fontsize=8.5, color=INK2)
    for mask, colour, label in [(~is_vb, BLUE, "Other signals"),
                                (is_vb, ORANGE, "Volatility / beta signals")]:
        ax.scatter(d.loc[mask, "train_ic"], d.loc[mask, "test_ic"], s=62, color=colour,
                   edgecolor=SURFACE, linewidth=1.6, label=label, zorder=3)

    # label only the points the story is about, with hand-set offsets to avoid overlap
    labels = {
        "mom_252_skip21": (8, -3, "left"), "rev_5": (8, -3, "left"),
        "amihud_21": (8, -4, "left"), "log_dvol_21": (-8, 4, "right"),
        "beta_126": (8, 3, "left"), "vol_21": (8, -1, "left"),
        "parkinson_21": (8, -9, "left"),
    }
    for name, (dx, dy, ha) in labels.items():
        if name in d.index:
            ax.annotate(name, (d.at[name, "train_ic"], d.at[name, "test_ic"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5,
                        color=INK2)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean rank IC, train 2017-2022")
    ax.set_ylabel("Mean rank IC, test 2023-2026")
    ax.legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.3,
              borderaxespad=0.2)
    rho = d["train_ic"].corr(d["test_ic"], method="spearman")
    titled(fig, "In-sample IC says little about out-of-sample IC",
           f"Each dot is one signal. Rank correlation {rho:.2f}; "
           f"sign agrees for {int(d['same_sign'].sum())} of {len(d)}")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.09)
    fig.savefig(OUT / "fig2_train_vs_test_ic.png", dpi=200)
    plt.close(fig)


def fig3_beta_hedge() -> None:
    d = pd.read_csv(DATA / "beta_check.csv", index_col="rule")
    names = {"ic_top5": "Top 5 by\nin-sample IC", "stab_top5": "Top 5 by\nstability",
             "all": "All 30,\nequal weight", "beta_126 only": "Long high beta\n(reference)"}
    d = d.loc[[r for r in names if r in d.index]]
    x = range(len(d))
    w = 0.3

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.axhline(0, color=INK2, linewidth=0.8)
    for off, col, colour, label in [(-w / 2 - 0.012, "ls_sharpe", BLUE, "As traded"),
                                    (w / 2 + 0.012, "hedged_sharpe", ORANGE,
                                     "After removing market exposure")]:
        bars = ax.bar([i + off for i in x], d[col], width=w, color=colour, label=label)
        for b, v in zip(bars, d[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.03), f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_xticks(list(x), [names[r] for r in d.index])
    ax.tick_params(axis="x", labelcolor=INK)
    ax.set_ylabel("Annualised Sharpe, test 2023-2026")
    ax.set_ylim(min(0, d[["ls_sharpe", "hedged_sharpe"]].min().min()) - 0.15,
                d[["ls_sharpe", "hedged_sharpe"]].max().max() + 0.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=2, borderaxespad=0.2)
    titled(fig, "Most of the out-of-sample Sharpe is market beta",
           "Top-minus-bottom quintile portfolios, gross of costs")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.80, bottom=0.14)
    fig.savefig(OUT / "fig3_beta_hedge.png", dpi=200)
    plt.close(fig)


def fig4_sharpe_intervals() -> bool:
    src = DATA / "backtest_summary.csv"
    if not src.exists():
        print("Skipping fig4: run scripts/07_backtest.py first.")
        return False
    d = pd.read_csv(src, index_col="rule")
    names = {"ic_top5": "Top 5 by\nin-sample IC", "stab_top5": "Top 5 by\nstability",
             "all": "All 30,\nequal weight", "beta_126 only": "Long high beta\n(reference)"}
    d = d.loc[[r for r in names if r in d.index]]
    y = list(range(len(d)))[::-1]                 # first rule at the top
    off = 0.16

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    ax.axvline(0, color=INK2, linewidth=0.8)
    for dy, pre, colour, label in [(off, "gross", BLUE, "As traded"),
                                   (-off, "hedged", ORANGE, "Hedged with ex-ante beta")]:
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.plot([r[f"{pre}_ci_lo"], r[f"{pre}_ci_hi"]], [yi + dy, yi + dy], color=colour,
                    linewidth=2.2, solid_capstyle="round", zorder=2)
            ax.text(r[f"{pre}_ci_hi"] + 0.06, yi + dy, f"{r[f'{pre}_sharpe']:.2f}", va="center",
                    fontsize=9)
        ax.scatter(d[f"{pre}_sharpe"], [yi + dy for yi in y], s=62, color=colour,
                   edgecolor=SURFACE, linewidth=1.6, label=label, zorder=3)
    ax.set_yticks(y, [names[r] for r in d.index])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_ylim(-0.6, len(d) - 0.25)
    lo, hi = d[["gross_ci_lo", "hedged_ci_lo"]].min().min(), d[["gross_ci_hi", "hedged_ci_hi"]].max().max()
    ax.set_xlim(lo - 0.15, hi + 0.45)
    ax.set_xlabel("Annualised Sharpe, test 2023-2026 (dot = estimate, line = 95% interval)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), frameon=False, fontsize=9, ncols=2,
              borderaxespad=0, handletextpad=0.3)
    titled(fig, "After hedging, no Sharpe is distinguishable from zero",
           "Block-bootstrap intervals over 187 weekly returns, gross of costs")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.13)
    fig.savefig(OUT / "fig4_sharpe_intervals.png", dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_selection_probability()
    fig2_train_vs_test_ic()
    fig3_beta_hedge()
    n = 3 + fig4_sharpe_intervals()
    print(f"Saved {n} figures to {OUT}")


if __name__ == "__main__":
    main()
