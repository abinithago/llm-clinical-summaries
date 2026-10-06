#!/usr/bin/env python3
"""Step 0 -- pull the source texts out of data/dataset.csv.

Writes data/notes.csv, one row per case, the input for steps 1 and 2:

    dataset (askdocs | medisumqa | acibench | mimic_bhc), context_id,
    clinical_context (full note), human_summary (ACI-Bench / MIMIC-IV BHC only)

Usage:
    python make_notes.py [--data data/dataset.csv] [--out data/notes.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DATASET_LABEL

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data/dataset.csv")
    ap.add_argument("--out", type=Path, default=HERE / "data/notes.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data, usecols=["dataset", "context_id", "condition",
                                         "clinical_context"], dtype={"context_id": str})
    df = df[df.condition.isin(["baseline", "human"])].dropna(subset=["clinical_context"])
    df = df.drop_duplicates(["dataset", "context_id", "condition"])
    wide = (df.pivot(index=["dataset", "context_id"], columns="condition",
                     values="clinical_context")
              .rename(columns={"baseline": "clinical_context", "human": "human_summary"})
              .reset_index())
    wide = wide.dropna(subset=["clinical_context"])
    wide["dataset"] = wide["dataset"].map({v: k for k, v in DATASET_LABEL.items()})
    cols = ["dataset", "context_id", "clinical_context"] + \
        (["human_summary"] if "human_summary" in wide else [])
    wide[cols].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(wide)} cases")
    print(wide.groupby("dataset").agg(cases=("context_id", "size"),
                                      human=("human_summary", "count")).to_string())


if __name__ == "__main__":
    main()
