"""
Step 4 - Out-of-sample check: does in-sample selection survive?

    python scripts/04_out_of_sample.py

Input : data/panel.parquet, data/stability_selection.csv, data/selected.json
Output: data/oos_signal_ic.csv     (train vs test IC for every signal)
        data/oos_composites.csv    (test performance of each selection rule)

Every choice (which signals, which sign) is made on the TRAIN period only.
The test period is used once, here, to score those choices.

Selection rules compared (K fixed in advance, not tuned):
  * ic_topK    - the naive approach: K signals with the largest |t-stat| of IC in train
  * stab_topK  - K signals with the highest stability-selection probability
  * all        - every signal, equal weight (no selection at all)
  * stable / lasso_cv - the sets from step 3, if non-empty

A composite is the equal-weight average of the chosen signals, each multiplied
by the sign of its train IC. It is scored out of sample by rank IC and by a
top-minus-bottom quintile portfolio (gross of costs).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

KS = (3, 5, 10)
PERIODS_PER_YEAR = 252 / 5      # panel is sampled every 5 trading days
DATA = Path("data")


def ic_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Spearman correlation of each column with the target, per date."""
    return df.groupby("date")[cols + ["target"]].apply(
        lambda g: g[cols].corrwith(g["target"], method="spearman")
    )


def ic_stats(ic: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ic": ic.mean(),
        "t": ic.mean() / ic.std() * np.sqrt(len(ic)),
    })


def long_short(df: pd.DataFrame, score: str) -> pd.Series:
    """Per-date return of top quintile minus bottom quintile on `score`."""
    def one(g):
        q = g[score].rank(pct=True)
        return g.loc[q > 0.8, "target"].mean() - g.loc[q <= 0.2, "target"].mean()
    return df.groupby("date")[[score, "target"]].apply(one)


def main() -> None:
    panel = pd.read_parquet(DATA / "panel.parquet")
    sel = json.loads((DATA / "selected.json").read_text())
    stab = pd.read_csv(DATA / "stability_selection.csv", index_col="signal")
    signals = sel["all"]

    train = panel[panel["date"] <= sel["train_end"]]
    test = panel[panel["date"] > sel["train_end"]].copy()

    # --- per-signal: does train IC carry over to test?
    tr, te = ic_stats(ic_by_date(train, signals)), ic_stats(ic_by_date(test, signals))
    per = tr.add_prefix("train_").join(te.add_prefix("test_"))
    per["stab_prob"] = stab["stab_prob"]
    per["same_sign"] = np.sign(per["train_ic"]) == np.sign(per["test_ic"])
    per = per.sort_values("train_t", key=np.abs, ascending=False)
    per.to_csv(DATA / "oos_signal_ic.csv")

    sign = np.sign(per["train_ic"])
    by_ic = per.index.tolist()                                    # already sorted by |train t|
    by_stab = per.sort_values("stab_prob", ascending=False).index.tolist()

    rules: dict[str, list[str]] = {}
    for k in KS:
        rules[f"ic_top{k}"] = by_ic[:k]
        rules[f"stab_top{k}"] = by_stab[:k]
    rules["all"] = signals
    rules["stable"] = sel["stable"]
    rules["lasso_cv"] = sel["lasso_cv"]

    rows = []
    for name, cols in rules.items():
        if not cols:
            rows.append({"rule": name, "n_signals": 0})
            continue
        test[name] = (test[cols] * sign[cols]).mean(axis=1)
        ic = ic_by_date(test, [name])[name]
        ls = long_short(test, name)
        rows.append({
            "rule": name,
            "n_signals": len(cols),
            "test_ic": ic.mean(),
            "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
            "ls_ann_ret": ls.mean() * PERIODS_PER_YEAR,
            "ls_sharpe": ls.mean() / ls.std() * np.sqrt(PERIODS_PER_YEAR),
            "hit_rate": (ls > 0).mean(),
        })
    comp = pd.DataFrame(rows).set_index("rule")
    comp.to_csv(DATA / "oos_composites.csv")

    print("Per-signal IC, train vs test (sorted by |train t|):")
    print(per.round(3).to_string())
    rank_corr = per["train_ic"].corr(per["test_ic"], method="spearman")
    print(
        f"\nTrain {train['date'].nunique()} dates, test {test['date'].nunique()} dates "
        f"({test['date'].min().date()} to {test['date'].max().date()})"
    )
    print(
        f"Spearman corr(train IC, test IC) across {len(signals)} signals = {rank_corr:.2f}; "
        f"sign agrees for {int(per['same_sign'].sum())}/{len(signals)}"
    )
    print("\nComposites, test period only (long-short is gross of costs):")
    print(comp.round(3).to_string())
    for k in KS:
        print(f"  ic_top{k}:   {rules[f'ic_top{k}']}")
        print(f"  stab_top{k}: {rules[f'stab_top{k}']}")


if __name__ == "__main__":
    main()
