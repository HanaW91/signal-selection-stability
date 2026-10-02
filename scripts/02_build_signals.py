"""
Step 2 - Build the candidate signal panel.

    python scripts/02_build_signals.py

Input : data/prices.parquet   (from step 1)
Output: data/panel.parquet    (date, ticker, 30 signals, target)
        data/signal_ic.csv    (sanity check: mean rank IC per signal)

Design choices (all worth a line in the README):
  * Every signal at date t uses data up to and including the close of t.
  * Target = return from close t+1 to close t+1+HORIZON, i.e. one day is skipped
    so nothing is traded on the same close the signal was computed from.
  * Target is cross-sectionally demeaned -> we predict relative performance,
    not market direction.
  * Signals are winsorised and z-scored within each date, so LASSO penalises
    them on a common scale and no date dominates.
  * Dates are sampled every HORIZON days so targets do not overlap.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 5          # trading days in the forward return
SKIP = 1             # days between signal date and start of the return window
WINSOR = 0.02        # clip each signal at the 2nd / 98th percentile per date
DATA = Path("data")


# ----------------------------------------------------------------- helpers
def wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long -> wide (index = date, columns = ticker)."""
    return prices.pivot(index="date", columns="ticker", values=col).sort_index()


def rolling_beta(r: pd.DataFrame, m: pd.Series, w: int) -> pd.DataFrame:
    """Rolling OLS beta of each column of r on the market return m."""
    cov = r.mul(m, axis=0).rolling(w).mean() - r.rolling(w).mean().mul(m.rolling(w).mean(), axis=0)
    return cov.div(m.rolling(w).var(ddof=0), axis=0)


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def xs_standardise(x: pd.DataFrame) -> pd.DataFrame:
    """Winsorise then z-score across tickers, separately for each date."""
    lo, hi = x.quantile(WINSOR, axis=1), x.quantile(1 - WINSOR, axis=1)
    x = x.clip(lower=lo, upper=hi, axis=0)
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


# ----------------------------------------------------------------- signals
def build_signals(o, h, l, c, v) -> dict[str, pd.DataFrame]:
    r = c.pct_change()
    mkt = r.mean(axis=1)                      # equal-weight market return
    dvol = c * v                              # dollar volume
    s: dict[str, pd.DataFrame] = {}

    # momentum / reversal
    s["mom_21"] = c.pct_change(21)
    s["mom_63"] = c.pct_change(63)
    s["mom_126"] = c.pct_change(126)
    s["mom_252_skip21"] = c.shift(21) / c.shift(252) - 1      # classic 12-1
    s["rev_1"] = -r
    s["rev_5"] = -c.pct_change(5)

    # trend / price level
    s["px_ma20"] = c / c.rolling(20).mean() - 1
    s["px_ma50"] = c / c.rolling(50).mean() - 1
    s["px_ma200"] = c / c.rolling(200).mean() - 1
    s["ma50_ma200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    s["dist_52w_high"] = c / c.rolling(252).max() - 1
    s["dist_52w_low"] = c / c.rolling(252).min() - 1

    # oscillators
    s["rsi_14"] = rsi(c, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    s["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    s["boll_z20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()

    # volatility / distribution shape
    s["vol_21"] = r.rolling(21).std()
    s["vol_63"] = r.rolling(63).std()
    s["vol_ratio_21_63"] = s["vol_21"] / s["vol_63"]
    s["downside_vol_63"] = r.clip(upper=0).rolling(63).std()
    s["parkinson_21"] = np.sqrt((np.log(h / l) ** 2).rolling(21).mean() / (4 * np.log(2)))
    s["max_ret_21"] = r.rolling(21).max()                     # lottery / MAX effect
    s["skew_63"] = r.rolling(63).skew()
    s["kurt_63"] = r.rolling(63).kurt()

    # market exposure
    beta = rolling_beta(r, mkt, 126)
    s["beta_126"] = beta
    s["idio_vol_63"] = (r - beta.mul(mkt, axis=0)).rolling(63).std()

    # volume / liquidity
    s["log_dvol_21"] = np.log(dvol.rolling(21).mean())
    s["abn_volume_5_63"] = v.rolling(5).mean() / v.rolling(63).mean() - 1
    s["amihud_21"] = np.log((r.abs() / dvol).rolling(21).mean())

    # intraday vs overnight
    s["overnight_21"] = (o / c.shift(1) - 1).rolling(21).sum()
    s["intraday_21"] = (c / o - 1).rolling(21).sum()

    return s


# -------------------------------------------------------------------- main
def main() -> None:
    prices = pd.read_parquet(DATA / "prices.parquet")
    o, h, l, c, v = (wide(prices, k) for k in ["open", "high", "low", "close", "volume"])
    v = v.where(v > 0)                         # zero volume -> NaN, avoids log(0) / div by 0

    signals = build_signals(o, h, l, c, v)
    signals = {k: xs_standardise(x.replace([np.inf, -np.inf], np.nan)) for k, x in signals.items()}

    # forward return from close t+SKIP to close t+SKIP+HORIZON, demeaned per date
    fwd = c.shift(-(SKIP + HORIZON)) / c.shift(-SKIP) - 1
    target = fwd.sub(fwd.mean(axis=1), axis=0)

    panel = pd.concat({k: x.stack() for k, x in signals.items()}, axis=1)
    panel["target"] = target.stack()
    panel = panel.dropna().reset_index()

    # keep every HORIZON-th date so the forward returns do not overlap
    keep = sorted(panel["date"].unique())[::HORIZON]
    panel = panel[panel["date"].isin(keep)].reset_index(drop=True)
    panel.to_parquet(DATA / "panel.parquet", index=False)

    # sanity check: rank IC = Spearman corr(signal, target) per date
    names = list(signals)
    ic = panel.groupby("date")[names + ["target"]].apply(
        lambda g: g[names].corrwith(g["target"], method="spearman")
    )
    summary = pd.DataFrame({
        "mean_ic": ic.mean(),
        "ic_std": ic.std(),
        "t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "pct_positive": (ic > 0).mean(),
    }).sort_values("t_stat", key=np.abs, ascending=False)
    summary.to_csv(DATA / "signal_ic.csv")

    print(summary.round(3).to_string())
    print(
        f"\nSaved panel: {len(panel):,} rows, {panel['date'].nunique()} dates, "
        f"{panel['ticker'].nunique()} tickers, {len(names)} signals, "
        f"{panel['date'].min().date()} to {panel['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
