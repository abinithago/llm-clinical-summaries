#!/usr/bin/env python3
"""Step 4 -- every metric reported for the study, from one long table.

Reads data/dataset.csv (or a rebuilt one from build_dataset.py) and writes CSVs
to results/:

  accuracy_pooled.csv        reader x condition, all four datasets, full note + LLM summaries
  accuracy_human_pair.csv    same, restricted to ACI-Bench + MIMIC-IV BHC (the only
                             datasets with a human-written summary), all conditions
  accuracy_by_dataset.csv    reader x dataset x condition
  accuracy_by_task.csv       reader x task x condition
  vs_full_note.csv           each summary condition against the full note, same reader:
                             delta, two-proportion z-test, Bonferroni within each table
  error_rates.csv            overburden (FPR: gold NO, answered YES) and harm
                             (FNR: gold YES, answered NO) per reader x task x condition
  physician_agreement.csv    agreement among the three physician reads per
                             condition x dataset: unanimity, pairwise, Fleiss' kappa

Accuracy intervals are Wilson intervals at alpha = 0.05 / (cells in that table),
which is how the paper figures draw them.

Usage:
    python compute_metrics.py [--data data/dataset.csv] [--out results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
PHYS = "Expert Physicians"
LLM_CONDS = ["baseline", "summary_gpt4o", "summary_llama", "summary_deepseek",
             "summary_medgemma"]
ALL_CONDS = ["baseline", "human", *LLM_CONDS[1:]]
HUMAN_DATASETS = ["ACI-Bench", "MIMIC-IV BHC"]


def wilson(k: float, n: float, alpha: float = 0.05):
    if not n:
        return np.nan, np.nan, np.nan
    p = k / n
    z = sps.norm.ppf(1 - alpha / 2)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def two_proportion_p(k1, n1, k2, n2) -> float:
    """Two-sided pooled-proportion z-test. Physician n counts reads (3 per
    case-decision) as independent, which understates its standard error."""
    if not (n1 > 0 and n2 > 0):
        return 1.0
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if not se > 0:
        return 1.0
    return float(2 * sps.norm.sf(abs((k1 / n1 - k2 / n2) / se)))


def ordered(df: pd.DataFrame, conds: list[str]) -> pd.DataFrame:
    df = df[df.condition.isin(conds)]
    return df.assign(_c=df.condition.map({c: i for i, c in enumerate(conds)}))


def accuracy(df: pd.DataFrame, keys: list[str], conds: list[str]) -> pd.DataFrame:
    df = ordered(df, conds)
    g = (df.groupby(keys + ["condition", "_c"], sort=False).correct
           .agg(k="sum", n="size").reset_index())
    alpha = 0.05 / max(len(g), 1)
    ci = pd.DataFrame([wilson(k, n, alpha) for k, n in zip(g.k, g.n)],
                      columns=["acc", "lo", "hi"])
    out = pd.concat([g, ci], axis=1).sort_values(keys + ["_c"]).drop(columns="_c")
    return out.reset_index(drop=True)


def vs_full_note(acc: pd.DataFrame, keys: list[str], table: str) -> pd.DataFrame:
    rows = []
    for k, grp in acc.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        base = grp[grp.condition == "baseline"]
        if base.empty:
            continue
        b = base.iloc[0]
        for r in grp[grp.condition != "baseline"].itertuples():
            rows.append({**dict(zip(keys, k)), "condition": r.condition,
                         "acc_full_note": b.acc, "acc_condition": r.acc,
                         "delta": r.acc - b.acc, "n_full_note": b.n, "n_condition": r.n,
                         "p_raw": two_proportion_p(r.k, r.n, b.k, b.n)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["m"] = len(out)
    out["p_bonferroni"] = np.minimum(1.0, out.p_raw * out.m)
    out["significant_05"] = out.p_bonferroni < 0.05
    out.insert(0, "table", table)
    return out


def error_rates(df: pd.DataFrame, conds: list[str]) -> pd.DataFrame:
    df = ordered(df, conds)
    rows = []
    for (reader, task, cond, c), g in df.groupby(["reader", "task", "condition", "_c"],
                                                  sort=False):
        for metric, base, wrong in (("FPR_overburden", "NO", "YES"),
                                    ("FNR_harm", "YES", "NO")):
            elig = g[g.gold == base]
            rows.append({"reader": reader, "task": task, "condition": cond, "_c": c,
                         "metric": metric, "k": int((elig.pred == wrong).sum()),
                         "n": len(elig)})
    out = pd.DataFrame(rows)
    alpha = 0.05 / max(len(out), 1)
    ci = pd.DataFrame([wilson(k, n, alpha) for k, n in zip(out.k, out.n)],
                      columns=["rate", "lo", "hi"])
    out = pd.concat([out, ci], axis=1).sort_values(["reader", "task", "_c", "metric"])
    out = out.drop(columns="_c").reset_index(drop=True)
    return out


def physician_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement of the three physician reads with each other (not with gold)."""
    p = df[(df.reader == PHYS) & df.physician_read.notna()]
    out = []
    for keys, s in [((c, d), g) for (c, d), g in p.groupby(["condition", "dataset"])] + \
                   [((c, "ALL"), g) for c, g in p.groupby("condition")]:
        w = (s.assign(one=1).pivot_table(index=["dataset", "context_id", "task"],
                                         columns="pred", values="one", aggfunc="sum",
                                         fill_value=0)
              .reindex(columns=["YES", "NO"], fill_value=0))
        w = w[w.sum(axis=1) == 3]
        if w.empty:
            continue
        counts = w.to_numpy(float)
        p_i = ((counts ** 2).sum(axis=1) - 3) / 6
        p_bar = p_i.mean()
        p_j = counts.sum(axis=0) / counts.sum()
        p_e = (p_j ** 2).sum()
        out.append({"condition": keys[0], "dataset": keys[1], "items": len(w),
                    "unanimous": (counts.max(axis=1) == 3).mean(), "pairwise": p_bar,
                    "fleiss_kappa": (p_bar - p_e) / (1 - p_e) if p_e < 1 else np.nan,
                    "yes_rate": p_j[0]})
    out = pd.DataFrame(out)
    if out.empty:
        return out
    out["_c"] = out.condition.map({c: i for i, c in enumerate(ALL_CONDS)})
    return out.sort_values(["dataset", "_c"]).drop(columns="_c").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data/dataset.csv")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    df = pd.read_csv(args.data, usecols=["reader", "dataset", "context_id", "condition",
                                         "task", "gold", "pred", "correct",
                                         "physician_read"])
    args.out.mkdir(parents=True, exist_ok=True)

    pair = df[df.dataset.isin(HUMAN_DATASETS)]
    tabs = {
        "accuracy_pooled": accuracy(df, ["reader"], LLM_CONDS),
        "accuracy_human_pair": accuracy(pair, ["reader"], ALL_CONDS),
        "accuracy_by_dataset": accuracy(df, ["reader", "dataset"], ALL_CONDS),
        "accuracy_by_task": accuracy(df, ["reader", "task"], ALL_CONDS),
        "error_rates": error_rates(df, ALL_CONDS),
        "physician_agreement": physician_agreement(df),
    }
    tabs["vs_full_note"] = pd.concat([
        vs_full_note(tabs["accuracy_pooled"], ["reader"], "pooled"),
        vs_full_note(tabs["accuracy_human_pair"], ["reader"], "human_pair"),
        vs_full_note(tabs["accuracy_by_dataset"], ["reader", "dataset"], "by_dataset"),
    ], ignore_index=True)

    for name, t in tabs.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        print(f"  {name}.csv  ({len(t)} rows)")

    show = tabs["accuracy_pooled"].pivot(index="reader", columns="condition", values="acc")
    print("\nPooled accuracy (VISIT + RESOURCE, all datasets)")
    print(show[[c for c in LLM_CONDS if c in show]].round(3).to_string())


if __name__ == "__main__":
    main()
