"""
Step 3 - Which signals are selected *stably*?  Stability selection vs plain LASSO.

    python scripts/03_stability_selection.py

Input : data/panel.parquet                 (from step 2)
Output: data/stability_selection.csv       (per-signal results)
        data/stability_paths.csv           (selection probability along the lambda path)
        data/selected.json                 (signal sets handed to step 4)

Everything here uses the TRAIN period only. The test period is not touched
until step 4, so the out-of-sample comparison stays honest.

Method
  * Stability selection (Meinshausen & Buhlmann, 2010): refit LASSO on B random
    half-samples, record how often each signal is selected, keep signals whose
    selection probability is >= PI_THR.
  * The resampling unit is the DATE, not the row. The ~100 stocks on one date
    are cross-sectionally dependent, so drawing rows would overstate how much
    independent data we have and make everything look more stable than it is.
  * The lambda range is capped so the average number of selected signals q
    satisfies the M&B bound  E[false selections] <= q^2 / ((2*PI_THR - 1) * p)
    <= PFER.
  * Baseline: LASSO with lambda chosen by date-grouped CV, refit on bootstrap
    resamples of dates, to show how much its selected set moves around.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.model_selection import GroupKFold

TRAIN_END = "2022-12-31"   # train <= this date, test after (used in step 4)
B = 200                    # half-samples for stability selection
PI_THR = 0.80              # selection-probability threshold
PFER = 1.0                 # tolerated expected number of false selections
N_LAMBDA = 40
N_BOOT = 50                # bootstrap refits of the LassoCV baseline
SEED = 42
DATA = Path("data")


def load_train():
    panel = pd.read_parquet(DATA / "panel.parquet")
    signals = [c for c in panel.columns if c not in ("date", "ticker", "target")]
    train = panel[panel["date"] <= TRAIN_END].reset_index(drop=True)
    return train, signals


def stability_selection(X, y, dates, rng):
    """Return lambda grid, index of lambda_min, selection-prob path, q at lambda_min."""
    n, p = X.shape
    lam_max = np.abs(X.T @ y).max() / n
    lambdas = np.geomspace(lam_max, lam_max * 0.01, N_LAMBDA)

    uniq = np.unique(dates)
    selected = np.zeros((B, N_LAMBDA, p), dtype=bool)
    for b in range(B):
        half = rng.choice(uniq, size=len(uniq) // 2, replace=False)
        m = np.isin(dates, half)
        _, coefs, _ = lasso_path(X[m], y[m], alphas=lambdas)
        selected[b] = (coefs != 0).T                      # (n_lambda, p)

    # q(lambda_k) = average size of the union of selected sets down to lambda_k
    union = np.logical_or.accumulate(selected, axis=1)
    q_path = union.sum(axis=2).mean(axis=0)
    q_max = np.sqrt((2 * PI_THR - 1) * p * PFER)
    ok = np.where(q_path <= q_max)[0]
    k = ok.max() if len(ok) else 0

    prob_path = selected.mean(axis=0)                     # (n_lambda, p)
    return lambdas, k, prob_path, q_path[k]


def lasso_cv_fit(X, y, dates):
    """LASSO with lambda picked by 5-fold CV, folds split by date."""
    cv = list(GroupKFold(n_splits=5).split(X, y, groups=dates))
    return LassoCV(cv=cv).fit(X, y)


def lasso_cv_bootstrap(X, y, dates, rng):
    """Refit LassoCV on bootstrap resamples of dates. Returns (N_BOOT, p) bool."""
    uniq = np.unique(dates)
    rows_by_date = {d: np.where(dates == d)[0] for d in uniq}
    out = np.zeros((N_BOOT, X.shape[1]), dtype=bool)
    for b in range(N_BOOT):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_date[d] for d in draw])
        out[b] = lasso_cv_fit(X[idx], y[idx], dates[idx]).coef_ != 0
    return out


def mean_jaccard(sets: np.ndarray) -> float:
    """Average pairwise Jaccard similarity between selected sets (1 = identical)."""
    vals = []
    for a, b in combinations(sets, 2):
        u = (a | b).sum()
        vals.append((a & b).sum() / u if u else 1.0)
    return float(np.mean(vals))


def main() -> None:
    rng = np.random.default_rng(SEED)
    train, signals = load_train()
    X = train[signals].to_numpy()
    y = train["target"].to_numpy()
    y = y / y.std()                                       # unit scale, readable lambdas
    dates = train["date"].to_numpy()
    p = len(signals)

    # --- stability selection
    lambdas, k, prob_path, q = stability_selection(X, y, dates, rng)
    stab_prob = prob_path[: k + 1].max(axis=0)
    stable = stab_prob >= PI_THR

    # --- plain LASSO baseline
    full = lasso_cv_fit(X, y, dates)
    boot = lasso_cv_bootstrap(X, y, dates, rng)
    sizes = boot.sum(axis=1)

    res = pd.DataFrame({
        "stab_prob": stab_prob,
        "stable": stable,
        "lasso_cv_coef": full.coef_,
        "lasso_boot_freq": boot.mean(axis=0),
    }, index=pd.Index(signals, name="signal")).sort_values(
        ["stab_prob", "lasso_boot_freq"], ascending=False
    )
    res.to_csv(DATA / "stability_selection.csv")

    paths = pd.DataFrame(prob_path, columns=signals)
    paths.insert(0, "lambda", lambdas)
    paths.insert(1, "in_range", np.arange(N_LAMBDA) <= k)
    paths.to_csv(DATA / "stability_paths.csv", index=False)

    stable_set = res.index[res["stable"]].tolist()
    lasso_set = res.index[res["lasso_cv_coef"] != 0].tolist()
    (DATA / "selected.json").write_text(json.dumps({
        "train_end": TRAIN_END,
        "stable": stable_set,
        "lasso_cv": lasso_set,
        "all": signals,
    }, indent=2))

    print(res.round(3).to_string())
    print(
        f"\nTrain: {len(train):,} rows, {len(np.unique(dates))} dates, "
        f"{pd.Timestamp(dates.min()).date()} to {pd.Timestamp(dates.max()).date()}"
    )
    print(
        f"Stability selection: B={B}, pi_thr={PI_THR}, lambda range uses "
        f"{k + 1}/{N_LAMBDA} grid points, q={q:.2f}, "
        f"PFER bound={q**2 / ((2 * PI_THR - 1) * p):.2f}"
    )
    print(f"  stable set ({len(stable_set)}): {stable_set}")
    print(f"LassoCV on full train ({len(lasso_set)}): {lasso_set}")
    print(
        f"LassoCV over {N_BOOT} bootstraps: set size min/median/max = "
        f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}, "
        f"mean pairwise Jaccard = {mean_jaccard(boot):.2f}"
    )


if __name__ == "__main__":
    main()
