"""
Step 7 - Backtest the step-4 portfolios properly: costs, ex-ante hedge, uncertainty.

    python scripts/07_backtest.py

Input : data/prices.parquet, data/panel.parquet, data/selected.json,
        data/stability_selection.csv
Output: data/backtest_summary.csv    (one row per rule)
        data/backtest_costs.csv      (net Sharpe at each cost level)
        data/backtest_sharpe_diff.csv (bootstrap CI for Sharpe differences)
        data/backtest_returns.csv    (per-date return series, for plotting)

Steps 4 and 5 left three things open, all listed under Limitations in the README:

  1. Costs. Returns were gross. Here each portfolio is held as explicit weights,
     turnover is measured at every rebalance, and returns are reported net of
     COST_BPS per unit of notional traded.
  2. The hedge used hindsight. Step 5 estimated one hedge ratio on the test
     period and applied it to the same data. Here the hedge ratio on date t is
     the portfolio's ex-ante beta: sum of weight x rolling 126-day beta, both
     known at the close of t.
  3. Sharpe ratios were point estimates. Here each one gets a block-bootstrap
     confidence interval, and the *difference* between rules is bootstrapped
     on the same resampled dates, which is the honest test of "is rule A
     better than rule B".

Portfolio: long the top quintile, short the bottom quintile, equal weight within
each leg, +1 / -1 gross per leg (dollar-neutral), rebalanced every 5 trading
days. Same construction as step 4, so the gross Sharpe must match it exactly.

Everything about which signals and which signs is still decided on TRAIN only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 5
HORIZON, SKIP = 5, 1
BETA_WINDOW = 126
PERIODS_PER_YEAR = 252 / HORIZON
COST_BPS = (0, 5, 10, 20)        # one-way cost per unit of notional traded
HEADLINE_COST = 10               # cost level used in the summary table
BLOCK = 4                        # block length in rebalance periods (~1 month)
N_BOOT = 5000
SEED = 42
DATA = Path("data")


# ----------------------------------------------------------------- helpers
def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    return x.mean() / x.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def ic_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("date")[cols + ["target"]].apply(
        lambda g: g[cols].corrwith(g["target"], method="spearman")
    )


def rolling_beta(r: pd.DataFrame, m: pd.Series, w: int) -> pd.DataFrame:
    """Rolling OLS beta of each column of r on m. Same formula as step 2."""
    cov = r.mul(m, axis=0).rolling(w).mean() - r.rolling(w).mean().mul(m.rolling(w).mean(), axis=0)
    return cov.div(m.rolling(w).var(ddof=0), axis=0)


def quintile_weights(score: pd.DataFrame) -> pd.DataFrame:
    """Wide score (date x ticker) -> weights: +1 across the top quintile, -1 across the bottom."""
    q = score.rank(axis=1, pct=True)
    long, short = (q > 0.8).astype(float), (q <= 0.2).astype(float)
    return long.div(long.sum(axis=1), axis=0) - short.div(short.sum(axis=1), axis=0)


def turnover(w: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """
    Notional traded at each rebalance. Yesterday's weights are first drifted by
    the returns earned over the holding period, so we only count real trades.
    The first date counts the cost of opening the book (= 2.0).
    """
    drifted = (w * (1 + fwd)).shift(1).fillna(0.0)
    return (w - drifted).abs().sum(axis=1)


def block_indices(n: int, rng) -> np.ndarray:
    """(N_BOOT, n) index matrix from a circular block bootstrap."""
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n
    return idx.reshape(N_BOOT, -1)[:, :n]


def boot_sharpe(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    s = x[idx]
    return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def ci(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# -------------------------------------------------------------------- main
def main() -> None:
    panel = pd.read_parquet(DATA / "panel.parquet")
    sel = json.loads((DATA / "selected.json").read_text())
    stab = pd.read_csv(DATA / "stability_selection.csv", index_col="signal")
    signals = sel["all"]

    train = panel[panel["date"] <= sel["train_end"]]
    test = panel[panel["date"] > sel["train_end"]].copy()
    dates = np.sort(test["date"].unique())

    # raw (not demeaned) forward returns and ex-ante betas, from prices
    prices = pd.read_parquet(DATA / "prices.parquet")
    c = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    r = c.pct_change()
    mkt_daily = r.mean(axis=1)
    fwd_all = c.shift(-(SKIP + HORIZON)) / c.shift(-SKIP) - 1
    fwd = fwd_all.reindex(dates)
    mkt = fwd_all.mean(axis=1).reindex(dates)             # same market series as step 5
    beta = rolling_beta(r, mkt_daily, BETA_WINDOW).reindex(dates)

    # same rules and signs as step 5, chosen on train only
    ic = ic_by_date(train, signals)
    t_train = ic.mean() / ic.std() * np.sqrt(len(ic))
    sign = np.sign(ic.mean())
    rules = {
        f"ic_top{K}": t_train.abs().sort_values(ascending=False).index[:K].tolist(),
        f"stab_top{K}": stab["stab_prob"].sort_values(ascending=False).index[:K].tolist(),
        "all": signals,
        "beta_126 only": ["beta_126"],
    }

    rng = np.random.default_rng(SEED)
    idx = block_indices(len(dates), rng)                  # shared by every rule -> paired comparisons
    m = mkt.to_numpy()

    series, summary, costs = {}, [], []
    for name, cols in rules.items():
        if name == "beta_126 only":
            test["_s"] = test["beta_126"]
        else:
            test["_s"] = (test[cols] * sign[cols]).mean(axis=1)
        score = test.pivot(index="date", columns="ticker", values="_s").reindex(dates)
        w = quintile_weights(score).fillna(0.0)

        gross = (w * fwd.reindex(columns=w.columns)).sum(axis=1)
        to = turnover(w, fwd.reindex(columns=w.columns).fillna(0.0))
        port_beta = (w * beta.reindex(columns=w.columns)).sum(axis=1)   # known at close of t
        hedged = gross - port_beta * mkt

        # in-sample hedge ratio, as in step 5, for comparison only
        b_insample = np.cov(gross, m, ddof=1)[0, 1] / m.var(ddof=1)
        resid_beta = np.cov(hedged, m, ddof=1)[0, 1] / m.var(ddof=1)

        net = {bps: gross - to * bps / 1e4 for bps in COST_BPS}
        net_hedged = hedged - to * HEADLINE_COST / 1e4

        g_lo, g_hi = ci(boot_sharpe(gross.to_numpy(), idx))
        h_lo, h_hi = ci(boot_sharpe(hedged.to_numpy(), idx))
        summary.append({
            "rule": name,
            "gross_sharpe": sharpe(gross),
            "gross_ci_lo": g_lo, "gross_ci_hi": g_hi,
            "turnover_per_rebal": to.iloc[1:].mean(),            # excludes opening the book
            "turnover_per_year": to.iloc[1:].mean() * PERIODS_PER_YEAR,
            "breakeven_bps": gross.mean() / to.mean() * 1e4,
            f"net_sharpe_{HEADLINE_COST}bps": sharpe(net[HEADLINE_COST]),
            "beta_insample": b_insample,
            "beta_exante_mean": port_beta.mean(),
            "hedged_sharpe": sharpe(hedged),
            "hedged_ci_lo": h_lo, "hedged_ci_hi": h_hi,
            "residual_beta": resid_beta,
            f"hedged_net_sharpe_{HEADLINE_COST}bps": sharpe(net_hedged),
        })
        costs.append({"rule": name, **{f"{bps}bps": sharpe(net[bps]) for bps in COST_BPS}})
        series[name] = pd.DataFrame({
            "gross": gross, "turnover": to, "port_beta": port_beta,
            "hedged": hedged, f"net_{HEADLINE_COST}bps": net[HEADLINE_COST],
        })

    summary = pd.DataFrame(summary).set_index("rule")
    costs = pd.DataFrame(costs).set_index("rule")

    # paired bootstrap of Sharpe differences: is rule A really better than rule B?
    pairs = [(f"stab_top{K}", f"ic_top{K}"), ("all", f"ic_top{K}"),
             (f"stab_top{K}", "beta_126 only"), ("all", "beta_126 only")]
    diffs = []
    for a, b in pairs:
        for kind in ("gross", "hedged"):
            xa, xb = series[a][kind].to_numpy(), series[b][kind].to_numpy()
            d = boot_sharpe(xa, idx) - boot_sharpe(xb, idx)
            lo, hi = ci(d)
            diffs.append({
                "a": a, "b": b, "returns": kind,
                "sharpe_diff": sharpe(xa) - sharpe(xb),
                "ci_lo": lo, "ci_hi": hi,
                "p_a_not_better": float((d <= 0).mean()),
            })
    diffs = pd.DataFrame(diffs)

    summary.to_csv(DATA / "backtest_summary.csv")
    costs.to_csv(DATA / "backtest_costs.csv")
    diffs.to_csv(DATA / "backtest_sharpe_diff.csv", index=False)
    pd.concat(series, axis=1).assign(mkt=mkt).to_csv(DATA / "backtest_returns.csv")

    print(f"Test period: {len(dates)} rebalances, "
          f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}\n")
    print("Summary (Sharpe annualised; CI = 95% circular block bootstrap, "
          f"block {BLOCK}, {N_BOOT} draws):")
    print(summary.round(3).T.to_string())
    print("\nNet Sharpe by one-way cost:")
    print(costs.round(3).to_string())
    print("\nSharpe differences, paired bootstrap:")
    print(diffs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
