#!/usr/bin/env python3
"""Step 3 -- collapse the per-seed decision files into one long table.

One row per (reader, dataset, context_id, condition, task), the schema of
data/dataset.csv:

    reader, dataset, context_id, condition, task, clinical_context,
    gold, pred, correct, physician_read

`pred` is the majority vote over seeds. Ties go to NO (votes are sorted before
counting, so the result never depends on set ordering). A case is kept for a
(reader, condition) only when every task has at least one valid vote, and only
when it has a gold label.

Expert Physicians rows in data/dataset.csv are not produced by this pipeline.
Pass --physician-from data/dataset.csv to carry them over unchanged so the
output is a drop-in replacement.

Usage:
    python build_dataset.py --out outputs/dataset.csv
    python build_dataset.py --out outputs/dataset.csv --physician-from data/dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DATASET_LABEL, MODELS

HERE = Path(__file__).resolve().parent
PHYS = "Expert Physicians"
# dataset key in this pipeline -> key used in data/gold_labels.csv
GOLD_KEY = {"askdocs": "askdocs", "medisumqa": "medisum",
            "acibench": "acibench", "mimic_bhc": "mimic_bhc"}
COND_ORDER = ["baseline", "human", "summary_gpt4o", "summary_llama",
              "summary_deepseek", "summary_medgemma"]


def majority(votes: list[str]) -> str:
    return max(sorted(set(votes)), key=votes.count)


def collect(dec_dir: Path, gold: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    gold = gold.set_index(["dataset", "context_id"])
    rows = []
    for ds_dir in sorted(p for p in dec_dir.iterdir() if p.is_dir()):
        ds = ds_dir.name
        for reader in sorted(p.name for p in ds_dir.iterdir() if p.is_dir()):
            if reader not in MODELS:
                continue
            for cdir in sorted(p for p in (ds_dir / reader).iterdir() if p.is_dir()):
                cond = cdir.name
                seeds = sorted(cdir.glob(f"{cond}_seed*.csv"))
                if not seeds:
                    continue
                frames = [pd.read_csv(f, dtype={"context_id": str}) for f in seeds]
                d = pd.concat(frames, ignore_index=True)
                texts = d.drop_duplicates("context_id").set_index("context_id")
                votes = {t: {} for t in tasks}
                for t in tasks:
                    for cid, v in zip(d["context_id"], d[f"llm_{t}"].astype(str)):
                        v = v.strip().upper()
                        if v in ("YES", "NO"):
                            votes[t].setdefault(cid, []).append(v)
                ids = set.intersection(*(set(v) for v in votes.values()))
                for cid in sorted(ids):
                    key = (GOLD_KEY.get(ds, ds), cid)
                    if key not in gold.index:
                        continue
                    g = gold.loc[key]
                    for t in tasks:
                        truth = str(g[f"gold_{t}"]).strip().upper()
                        if truth not in ("YES", "NO"):
                            continue
                        pred = majority(votes[t][cid])
                        rows.append({
                            "reader": MODELS[reader][2],
                            "dataset": DATASET_LABEL.get(ds, ds),
                            "context_id": cid, "condition": cond, "task": t.upper(),
                            "clinical_context": texts.at[cid, "clinical_context"],
                            "gold": truth, "pred": pred, "correct": int(pred == truth),
                            "physician_read": pd.NA,
                        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", type=Path, default=HERE / "outputs/decisions")
    ap.add_argument("--gold", type=Path, default=HERE / "data/gold_labels.csv")
    ap.add_argument("--tasks", nargs="+", default=["visit", "resource"],
                    choices=["manage", "visit", "resource"])
    ap.add_argument("--physician-from", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "outputs/dataset.csv")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold, dtype={"context_id": str})
    df = collect(args.decisions_dir, gold, args.tasks)
    if args.physician_from:
        phys = pd.read_csv(args.physician_from, dtype={"context_id": str})
        df = pd.concat([phys[phys.reader == PHYS], df], ignore_index=True)

    rank = {r: i for i, r in enumerate([PHYS] + [m[2] for m in MODELS.values()])}
    df = df.assign(_r=df.reader.map(rank),
                   _c=df.condition.map({c: i for i, c in enumerate(COND_ORDER)}),
                   _t=df.task.map({"MANAGE": 0, "VISIT": 1, "RESOURCE": 2}))
    df = df.sort_values(["_r", "dataset", "context_id", "_c", "_t", "physician_read"],
                        na_position="last").drop(columns=["_r", "_c", "_t"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
