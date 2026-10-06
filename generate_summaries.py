#!/usr/bin/env python3
"""Step 1 -- summarize each full note with one generator model.

Input CSV needs: dataset (askdocs | medisumqa | acibench | mimic_bhc),
context_id, clinical_context (the full note). Writes one file per dataset:

    <out-dir>/<generator>_<dataset>.csv   columns: dataset, context_id, clinical_context, thinking_leak

`thinking_leak` is True when a thinking model ran out of tokens before closing
its reasoning block, so the "summary" is raw reasoning. Re-run those rows with a
larger --max-new-tokens (MedGemma on long MIMIC/MeDiSumQA notes needs ~4096).

Usage:
    python generate_summaries.py --generator llama --input notes.csv
    python generate_summaries.py --generator gpt4o --input notes.csv --resume
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import (DATASETS, MODELS, SUMMARY_SYSTEM_MSG, is_thinking_model,
                    load_backend, strip_thinking, summary_prompt)

HERE = Path(__file__).resolve().parent

# GPT-4o settings used for the released summaries (temperature 0.1 throughout).
AZURE_MAX_TOKENS = {"askdocs": 512, "medisumqa": 512, "acibench": 512, "mimic_bhc": 1024}
TEMPERATURE = 0.1


def leaked(raw: str, model_id: str) -> bool:
    """True when the reasoning block never closed, so no real summary followed.

    Deepseek-R1 always reasons but never prints the opening tag, so a
    missing </think> is the signal. MedGemma only sometimes thinks, so it counts
    only when an opened <unused94> block is left unclosed."""
    m = model_id.lower()
    if "deepseek" in m:
        return "</think>" not in raw
    if "medgemma" in m:
        return "<unused94>" in raw and "<unused95>" not in raw
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, choices=list(MODELS))
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs/summaries")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="HF models: default 2048 for thinking models, else 512")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip context_ids already summarized (leaked rows are redone)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["dataset"] = df["dataset"].str.lower()
    df = df[df["dataset"].isin(args.datasets)].drop_duplicates(["dataset", "context_id"])
    if args.max_rows:
        df = df.head(args.max_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ds: args.out_dir / f"{args.generator}_{ds}.csv" for ds in args.datasets}
    done = {ds: pd.DataFrame() for ds in args.datasets}
    if args.resume:
        for ds, p in paths.items():
            if p.exists():
                d = pd.read_csv(p)
                done[ds] = d[~d["thinking_leak"].astype(bool)]
    done_keys = {(ds, str(c)) for ds, d in done.items() if len(d) for c in d["context_id"]}
    todo = df[[(ds, str(c)) not in done_keys for ds, c in zip(df.dataset, df.context_id)]]
    print(f"{args.generator}: {len(todo)} notes to summarize")
    if todo.empty:
        return

    backend = MODELS[args.generator][0]
    chat, model_id = load_backend(args.generator)
    max_new = args.max_new_tokens or (2048 if is_thinking_model(model_id) else 512)

    new = {ds: [] for ds in args.datasets}
    for r in tqdm(todo.itertuples(), total=len(todo)):
        messages = [{"role": "system", "content": SUMMARY_SYSTEM_MSG},
                    {"role": "user", "content": summary_prompt(r.dataset, str(r.clinical_context),
                                                               args.generator)}]
        try:
            if backend == "azure":
                raw = chat(messages, max_tokens=AZURE_MAX_TOKENS[r.dataset],
                           temperature=TEMPERATURE)
            else:
                raw = chat(messages, max_new_tokens=max_new, temperature=TEMPERATURE)
        except Exception as e:  # keep going; the row is marked and can be redone
            print(f"  ERROR {r.context_id}: {e}")
            raw = f"[ERROR: {e}]"
        text = re.sub(r"^Summary:\s*", "", strip_thinking(raw), flags=re.IGNORECASE)
        new[r.dataset].append({"dataset": r.dataset, "context_id": r.context_id,
                               "clinical_context": text,
                               "thinking_leak": raw.startswith("[ERROR") or leaked(raw, model_id)})

    for ds, p in paths.items():
        out = pd.concat([done[ds], pd.DataFrame(new[ds])], ignore_index=True)
        if len(out):
            out.to_csv(p, index=False)
            print(f"  {p.name}: {len(out)} rows, {int(out.thinking_leak.sum())} leaked")


if __name__ == "__main__":
    main()
