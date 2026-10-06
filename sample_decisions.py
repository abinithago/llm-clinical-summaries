#!/usr/bin/env python3
"""Step 2 -- ask a reader model the MANAGE / VISIT / RESOURCE questions.

Every condition is one version of the same cases:

    baseline          the full note           (--notes, column clinical_context)
    human             human-written summary   (--notes, column human_summary, if present)
    summary_<gen>     a model summary         (<summaries-dir>/<gen>_<dataset>.csv, from step 1)

Writes one file per (dataset, reader, condition, seed):

    <out-dir>/<dataset>/<reader>/<condition>/<condition>_seed<N>.csv
    columns: dataset, context_id, condition, clinical_context (the text the reader
             saw), llm_raw_output, llm_manage, llm_visit, llm_resource,
             llm_resource_specification

Re-running skips rows that already have YES/NO for all three tasks, so the same
command both resumes an interrupted job and repairs rows that failed to parse.

Usage:
    python sample_decisions.py --reader llama --notes notes.csv
    python sample_decisions.py --reader gpt4o --notes notes.csv --datasets acibench --seeds 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import DATASETS, MODELS, TASKS, decision_messages, load_backend, parse_decision

HERE = Path(__file__).resolve().parent
OUT_COLS = ["dataset", "context_id", "condition", "clinical_context", "llm_raw_output",
            *[f"llm_{t}" for t in TASKS], "llm_resource_specification"]


def condition_inputs(notes: pd.DataFrame, summaries_dir: Path, dataset: str,
                     generators: list[str]) -> dict[str, pd.DataFrame]:
    """condition -> frame(context_id, clinical_context) for one dataset."""
    n = notes[notes["dataset"] == dataset]
    out = {"baseline": n[["context_id", "clinical_context"]]}
    if "human_summary" in n and n["human_summary"].notna().any():
        out["human"] = (n[["context_id", "human_summary"]].dropna()
                        .rename(columns={"human_summary": "clinical_context"}))
    for g in generators:
        p = summaries_dir / f"{g}_{dataset}.csv"
        if not p.exists():
            print(f"  [skip] no summaries at {p}")
            continue
        s = pd.read_csv(p)
        if "thinking_leak" in s:
            bad = int(s["thinking_leak"].astype(bool).sum())
            if bad:
                print(f"  [warn] {p.name}: {bad} rows are leaked reasoning, excluded")
            s = s[~s["thinking_leak"].astype(bool)]
        out[f"summary_{g}"] = s[["context_id", "clinical_context"]]
    return {c: d.drop_duplicates("context_id").astype({"context_id": str})
            for c, d in out.items()}


def complete(d: pd.DataFrame) -> pd.Series:
    return d[[f"llm_{t}" for t in TASKS]].isin(["YES", "NO"]).all(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=list(MODELS))
    ap.add_argument("--notes", required=True, type=Path,
                    help="CSV: dataset, context_id, clinical_context[, human_summary]")
    ap.add_argument("--summaries-dir", type=Path, default=HERE / "outputs/summaries")
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs/decisions")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--generators", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict to these conditions, e.g. baseline summary_llama")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    notes = pd.read_csv(args.notes)
    notes["dataset"] = notes["dataset"].str.lower()
    notes["context_id"] = notes["context_id"].astype(str)

    backend, model_id, _ = MODELS[args.reader]
    chat = None  # loaded on first real work, so a no-op resume costs nothing

    for ds in args.datasets:
        conds = condition_inputs(notes, args.summaries_dir, ds, args.generators)
        for cond, inp in conds.items():
            if args.conditions and cond not in args.conditions:
                continue
            if args.max_rows:
                inp = inp.head(args.max_rows)
            for seed in args.seeds:
                path = args.out_dir / ds / args.reader / cond / f"{cond}_seed{seed}.csv"
                prev = pd.read_csv(path, dtype={"context_id": str}) if path.exists() else \
                    pd.DataFrame(columns=OUT_COLS)
                ok = set(prev.loc[complete(prev), "context_id"]) if len(prev) else set()
                todo = inp[~inp["context_id"].isin(ok)]
                if todo.empty:
                    continue
                if chat is None:
                    chat, model_id = load_backend(args.reader)
                random.seed(seed)
                if hasattr(chat, "torch"):
                    chat.torch.manual_seed(seed)

                rows = []
                desc = f"{args.reader} {ds}/{cond} seed{seed}"
                for r in tqdm(todo.itertuples(), total=len(todo), desc=desc):
                    msgs = decision_messages(str(r.clinical_context), model_id)
                    try:
                        if backend == "azure":
                            raw = chat(msgs, max_tokens=256, temperature=0.5)
                        elif "deepseek" in model_id.lower():
                            raw = chat(msgs, max_new_tokens=2048, temperature=None)
                        else:
                            raw = chat(msgs, max_new_tokens=1024, temperature=0.1)
                    except Exception as e:
                        print(f"  ERROR {r.context_id}: {e}")
                        raw = f"[ERROR: {e}]"
                    p = parse_decision(raw)
                    rows.append({"dataset": ds, "context_id": r.context_id, "condition": cond,
                                 "clinical_context": r.clinical_context,
                                 "llm_raw_output": raw,
                                 **{f"llm_{t}": p[t] for t in TASKS},
                                 "llm_resource_specification": p["resource_specification"]})

                new = pd.DataFrame(rows)
                out = pd.concat([prev[~prev["context_id"].isin(new["context_id"])], new],
                                ignore_index=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                out[OUT_COLS].to_csv(path, index=False)
                print(f"  {path.relative_to(args.out_dir)}: "
                      f"{int(complete(out).sum())}/{len(inp)} complete")


if __name__ == "__main__":
    main()
