"""
Step 1 - Download daily OHLCV for ~100 large-cap S&P 500 names.

Run from the repo root:
    pip install yfinance pandas pyarrow
    python scripts/01_download_prices.py

Output: data/prices.parquet  (long format: date, ticker, open, high, low, close, volume)
        data/universe.csv    (tickers kept after the coverage filter)

Known limitation (put this in the README): the universe is today's large caps,
so there is survivorship bias. Fine for a study about selection *stability*,
not fine for claiming a tradable return.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

START = "2016-01-01"
END = "2026-09-30"
MIN_COVERAGE = 0.98  # drop tickers missing more than 2% of trading days
OUT_DIR = Path("data")

TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "INTC", "AMD",
    "QCOM", "TXN", "IBM", "AMAT", "MU", "LRCX", "ADI", "KLAC", "INTU", "NOW",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "GM", "F",
    # Consumer staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "TGT",
    # Health care
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "BMY", "GILD", "MDT", "CVS", "ISRG",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "USB",
    "PNC", "V", "MA", "SPGI", "CB",
    # Industrials
    "CAT", "BA", "HON", "UNP", "UPS", "GE", "LMT", "RTX", "DE", "MMM", "FDX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities / real estate / materials
    "NEE", "DUK", "SO", "AMT", "PLD", "APD", "SHW",
]


def download(tickers: list[str]) -> pd.DataFrame:
    """Return long-format OHLCV, prices adjusted for splits and dividends."""
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )
    frames, failed = [], []
    have = set(raw.columns.get_level_values(0))
    for t in tickers:
        df = raw[t].dropna(how="all") if t in have else pd.DataFrame()
        if df.empty:
            failed.append(t)
            continue
        df = df.rename(columns=str.lower).rename_axis("date").reset_index()
        df["ticker"] = t
        frames.append(df)
    if failed:
        print(f"No data for {len(failed)} tickers: {', '.join(failed)}")
    if not frames:
        raise SystemExit("Nothing downloaded - check your connection and try again.")
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]


def filter_coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep tickers that trade on at least MIN_COVERAGE of all dates."""
    n_dates = prices["date"].nunique()
    coverage = prices.groupby("ticker")["close"].count() / n_dates
    dropped = coverage[coverage < MIN_COVERAGE]
    if len(dropped):
        print("Dropped for low coverage:")
        print(dropped.round(3).to_string())
    keep = coverage[coverage >= MIN_COVERAGE].index
    return prices[prices["ticker"].isin(keep)].reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    prices = filter_coverage(download(TICKERS))

    prices.to_parquet(OUT_DIR / "prices.parquet", index=False)
    pd.Series(sorted(prices["ticker"].unique()), name="ticker").to_csv(
        OUT_DIR / "universe.csv", index=False
    )

    print(
        f"\nSaved {len(prices):,} rows, "
        f"{prices['ticker'].nunique()} tickers, "
        f"{prices['date'].min().date()} to {prices['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
