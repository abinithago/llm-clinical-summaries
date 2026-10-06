# Summarization study

Do clinical decisions change when the decision-maker reads a summary instead of the full note?
Four LLM readers answer two triage questions on four corpora. Each reader sees the full note, a
human-written summary (where one exists), and summaries written by four LLMs.

This directory is self-contained. It holds the released decision-level data, the scripts that
generate summaries and sample the readers, and the script that computes every reported metric.

## Data use

`dataset.csv` and `notes.csv` contain MIMIC-IV note text, which falls under the
PhysioNet credentialed data use agreement. Keep the MIMIC rows off public repositories and
public hosting.

## Layout

```
summarization/
├── data/
│   ├── dataset.csv          released decision-level table (copy of codebase/figures/accuracy/dataset.csv)
│   ├── gold_labels.csv      per-case gold labels (VISIT, RESOURCE, MANAGE)
│   └── notes.csv            full notes + human summaries, extracted by make_notes.py
├── common.py                models, prompts, response parser, Azure / HuggingFace backends
├── summary_prompts_gpt4o.py prompts GPT-4o used for AskDocs + MeDiSumQA summaries (verbatim copy)
├── make_notes.py            step 0  dataset.csv -> notes.csv
├── generate_summaries.py    step 1  notes -> one summary per case per generator
├── sample_decisions.py      step 2  each reader x condition x seed -> YES/NO decisions
├── build_dataset.py         step 3  majority vote over seeds -> long table (dataset.csv schema)
├── compute_metrics.py       step 4  dataset.csv -> results/*.csv
└── results/                 output of compute_metrics.py on data/dataset.csv
```

## `data/dataset.csv`

83,938 rows; one row per reader × dataset × case × condition × task.

| column | meaning |
|---|---|
| `reader` | `GPT-4o`, `Llama-3.3-70B`, `Deepseek-R1`, `MedGemma-27B`, or `Expert Physicians` |
| `dataset` | `AskDocs` (193 cases), `MeDiSumQA` (377), `ACI-Bench` (207), `MIMIC-IV BHC` (500) |
| `context_id` | case id |
| `condition` | `baseline` (full note), `human` (ACI-Bench SOAP note / MIMIC brief hospital course), `summary_<generator>` |
| `task` | `VISIT` (come into clinic/urgent care/ED?) or `RESOURCE` (order a lab, test, imaging or referral?) |
| `clinical_context` | the exact text the reader was shown |
| `gold`, `pred` | `YES` / `NO`. For LLMs, `pred` is the majority vote over 3 seeds, with ties counted as `NO` |
| `correct` | `pred == gold` |
| `physician_read` | 0, 1, 2 for the three physician reads; empty for LLM rows |

Physician rows cover `baseline`, `human`, `summary_gpt4o` and `summary_llama` only.

## Models

| key | model | used as | decoding (decisions) |
|---|---|---|---|
| `gpt4o` | `gpt-4o` (Azure) | generator + reader | temperature 0.5, 256 tokens |
| `llama` | `meta-llama/Llama-3.3-70B-Instruct` | generator + reader | temperature 0.1, 1024 tokens |
| `deepseek` | `Deepseek-R1` | generator + reader | greedy, 2048 tokens |
| `medgemma` | `google/medgemma-27b-text-it` | generator + reader | temperature 0.1, 1024 tokens |

Summaries use temperature 0.1. HuggingFace generators use repetition penalty 1.1 and 512 new tokens,
or 2048 for the thinking models DeepSeek and MedGemma. GPT-4o uses 512 tokens, or 1024 on MIMIC.
Every reader gets the same decision prompt, `common.decision_prompt`. It also asks a MANAGE
question, but the released table and metrics use only VISIT and RESOURCE.

## Reproducing

```bash
pip install -r requirements.txt
export AZURE_OPENAI_API_KEY=...  AZURE_OPENAI_ENDPOINT=...   # GPT-4o only

python make_notes.py                                  # data/notes.csv
for g in gpt4o llama deepseek medgemma; do            # 4 generators
  python generate_summaries.py --generator $g --input data/notes.csv
done
for r in gpt4o llama deepseek medgemma; do            # 4 readers, 3 seeds each
  python sample_decisions.py --reader $r --notes data/notes.csv
done
python build_dataset.py --physician-from data/dataset.csv --out outputs/dataset.csv
python compute_metrics.py --data outputs/dataset.csv --out outputs/results
```

All HuggingFace models load in bf16 with `device_map="auto"`. The 70B Llama needs about 2×80 GB GPUs.
Both step 1 and step 2 resume when re-run:

- `generate_summaries.py --resume` skips cases that are already done. It redoes any row whose
  `thinking_leak` is True, meaning the model hit its token cap mid-reasoning and no summary was
  written. MedGemma on long MIMIC and MeDiSumQA notes needs `--max-new-tokens 4096`.
- `sample_decisions.py` only reruns rows that lack a YES/NO for every task, so rerunning the
  same command repairs failed parses.

`build_dataset.py` does not regenerate Expert Physicians rows. `--physician-from` copies
them unchanged.

## Metrics (`compute_metrics.py`)

| file | content |
|---|---|
| `accuracy_pooled.csv` | reader × condition, all datasets, full note + 4 LLM summaries |
| `accuracy_human_pair.csv` | ACI-Bench + MIMIC only, with the human summary added |
| `accuracy_by_dataset.csv`, `accuracy_by_task.csv` | the same, split |
| `vs_full_note.csv` | each summary vs. the full note for the same reader: delta, pooled two-proportion z-test, Bonferroni within each table |
| `error_rates.csv` | overburden = FPR (gold NO, answered YES); harm = FNR (gold YES, answered NO) |
| `physician_agreement.csv` | unanimity, pairwise agreement and Fleiss' κ among the 3 physician reads |

Intervals are Wilson intervals at α = 0.05 / (number of cells in the table). For physician rows,
n counts each read as independent, which understates their standard error.

On `data/dataset.csv`, `accuracy_pooled.csv` and `accuracy_human_pair.csv` match
`codebase/figures/accuracy/reader_comparison_accuracy{,_human}.csv` exactly (k, n, accuracy and
interval). Two outputs differ from the figure CSVs by construction:

- `error_rates.csv` has the same point estimates as `reader_comparison_rates.csv`. Its intervals
  are slightly wider (≤ 0.0013) because the human condition joins the Bonferroni family.
- `physician_agreement.csv` is computed over VISIT and RESOURCE only. The figure version also
  includes MANAGE, so its κ differs by up to 0.04.

Pooled accuracy, VISIT + RESOURCE:

| reader | full note | GPT-4o sum. | Llama sum. | DeepSeek sum. | MedGemma sum. |
|---|---|---|---|---|---|
| GPT-4o | 0.803 | 0.754 | 0.735 | 0.725 | 0.717 |
| Llama-3.3-70B | 0.745 | 0.718 | 0.704 | 0.716 | 0.732 |
| Deepseek-R1 | 0.654 | 0.593 | 0.594 | 0.624 | 0.656 |
| MedGemma-27B | 0.670 | 0.657 | 0.652 | 0.658 | 0.645 |
| Expert Physicians | 0.895 | 0.697 | 0.633 | — | — |
