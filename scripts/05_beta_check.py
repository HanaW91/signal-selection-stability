"""
Step 5 - Is the out-of-sample performance alpha, or just market beta?

    python scripts/05_beta_check.py

Input : data/prices.parquet, data/panel.parquet, data/selected.json,
        data/stability_selection.csv
Output: data/beta_check.csv

Why: the long-short portfolios in step 4 are dollar-neutral but not
beta-neutral. A composite that loads on volatility / beta signals is long
high-beta names and short low-beta names, so it earns the market premium in a
rising market. Here each test-period long-short series is regressed on the
market return over the same 5-day windows:

    LS_t = alpha + beta * MKT_t + e_t

and we report what is left (alpha) once the market exposure is removed.
MKT is the equal-weight return of the universe, same skip and horizon as the target.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 5
HORIZON, SKIP = 5, 1
PERIODS_PER_YEAR = 252 / HORIZON
DATA = Path("data")


def ic_by_date(df, cols):
    return df.groupby("date")[cols + ["target"]].apply(
        lambda g: g[cols].corrwith(g["target"], method="spearman")
    )


def long_short(df, score):
    def one(g):
        q = g[score].rank(pct=True)
        return g.loc[q > 0.8, "target"].mean() - g.loc[q <= 0.2, "target"].mean()
    return df.groupby("date")[[score, "target"]].apply(one)


def ols(y: np.ndarray, x: np.ndarray):
    """y = a + b x. Returns a, b, t(a)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return coef[0], coef[1], coef[0] / se[0]


def main() -> None:
    panel = pd.read_parquet(DATA / "panel.parquet")
    sel = json.loads((DATA / "selected.json").read_text())
    stab = pd.read_csv(DATA / "stability_selection.csv", index_col="signal")
    signals = sel["all"]

    train = panel[panel["date"] <= sel["train_end"]]
    test = panel[panel["date"] > sel["train_end"]].copy()

    # market return over the same forward window as the target
    prices = pd.read_parquet(DATA / "prices.parquet")
    c = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    mkt = (c.shift(-(SKIP + HORIZON)) / c.shift(-SKIP) - 1).mean(axis=1)

    ic = ic_by_date(train, signals)
    t_train = ic.mean() / ic.std() * np.sqrt(len(ic))
    sign = np.sign(ic.mean())
    rules = {
        f"ic_top{K}": t_train.abs().sort_values(ascending=False).index[:K].tolist(),
        f"stab_top{K}": stab["stab_prob"].sort_values(ascending=False).index[:K].tolist(),
        "all": signals,
        "beta_126 only": ["beta_126"],
    }

    rows = []
    for name, cols in rules.items():
        if name == "beta_126 only":      # reference portfolio: simply long high beta
            test["_s"] = test["beta_126"]
        else:
            test["_s"] = (test[cols] * sign[cols]).mean(axis=1)
        ls = long_short(test, "_s")
        m = mkt.reindex(ls.index)
        a, b, ta = ols(ls.to_numpy(), m.to_numpy())
        hedged = ls - b * m
        rows.append({
            "rule": name,
            "ls_sharpe": ls.mean() / ls.std() * np.sqrt(PERIODS_PER_YEAR),
            "beta_to_mkt": b,
            "corr_with_mkt": np.corrcoef(ls, m)[0, 1],
            "alpha_ann": a * PERIODS_PER_YEAR,
            "alpha_t": ta,
            "hedged_sharpe": hedged.mean() / hedged.std() * np.sqrt(PERIODS_PER_YEAR),
        })
    out = pd.DataFrame(rows).set_index("rule")
    out.to_csv(DATA / "beta_check.csv")

    m = mkt.reindex(test["date"].unique())
    print(
        f"Test period market (equal-weight universe): "
        f"{m.mean() * PERIODS_PER_YEAR:.1%} per year, "
        f"Sharpe {m.mean() / m.std() * np.sqrt(PERIODS_PER_YEAR):.2f}\n"
    )
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
