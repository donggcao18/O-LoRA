# O-LoRA — Code Task Mode (single task)

This repository originally trains UIE-style instruction data. It has been extended with a **`dataset_mode=code`** that lets you fine-tune the same LoRA pipeline on **code tasks from HuggingFace Datasets**, **one task per run**.

Supported tasks (current):
- `CodeTrans` (code translation)
- `CoST` (code translation)
- `CONCODE` (code generation)
- `BFP` (code refinement)
- `KodCode` (code generation, train-only split)
- `RunBugRun` (code refinement, train-only split)
- `CodeSearchNet` (code summarization)
- `TheVault_Csharp` (code summarization)

The dataset code lives in `src/code_tasks_dataset.py` and instruction templates/pools are in `src/task_info.py`.

---

## 1) Environment (Windows)

### 1.1 Create an environment
Use **Python 3.10** (recommended).

### 1.2 Install dependencies
Install the Python dependencies from `requirements.txt`.

> Note: `requirements.txt` does **not** pin `torch`. Install a CUDA-enabled PyTorch build first if you want GPU training.

---

## 2) Model: T5-Large

You can use:
- a local path, e.g. `initial_model/t5-large`
- or a HF model id: `t5-large`

On Windows, a **local** folder is usually more reliable.

---

## 3) Quick sanity run (small subset)

This run uses:
- `dataset_mode=code`
- `code_task=CodeTrans`
- subset size `code_k=200` for train/valid/test

Recommended output folder is short to avoid Windows path/lock issues.

Example arguments (fill in a real output dir):
- `--model_name_or_path initial_model/t5-large`
- `--dataset_mode code --code_task CodeTrans --code_k 200`
- `--do_train --do_eval --do_predict`
- `--predict_with_generate`

Common stable settings for a quick test:
- `--per_device_train_batch_size 2`
- `--per_device_eval_batch_size 2`
- `--learning_rate 2e-4`
- `--num_train_epochs 1`

---

## 4) Full training run

Set `--code_k -1` (default) to use the full dataset splits.

---

## 5) Metrics

When `dataset_mode=code`, evaluation metrics are BLEU-based:

- For **all tasks except** `CodeSearchNet` and `TheVault_Csharp`: `bleu_for_{task}`
- For `CodeSearchNet` and `TheVault_Csharp`: `smooth_bleu_for_{task}`

Implementation:
- metrics selection: `src/run_uie_lora.py` → `compute_code_metrics()`
- smooth BLEU implementation: `src/smooth_bleu_utils.py`

Metrics are reported under the trainer prefix, e.g.:
- `eval_bleu_for_CodeTrans`
- `eval_smooth_bleu_for_CodeSearchNet`
- `predict_bleu_for_CONCODE`

---

## 6) Output

The training output directory contains:
- trainer logs and metrics JSON
- LoRA adapter weights under `output_dir/adapter`

---

## 7) Troubleshooting

### 7.1 Windows cache / FileLock errors
This repo uses a short HF cache directory by default:
- `${drive}\hf_cache`

You can override by passing `--cache_dir`.

### 7.2 Dataset splits missing
Some tasks are train-only (e.g. `KodCode`, `RunBugRun`). In that case we split the train set into train/validation/test inside `build_code_task_dataset()`.

---

## 8) Minimal command template

Run from the repo root:

- `python src/run_uie_lora.py [ModelArguments] [DataTrainingArguments] [UIETrainingArguments]`

Essential flags for code mode:
- `--model_name_or_path ...`
- `--dataset_mode code`
- `--code_task <TASK>`
- `--output_dir <DIR>`
- `--do_train/--do_eval/--do_predict`
- `--predict_with_generate`

Optional flags:
- `--code_k <N>`: subset size per split
- `--max_source_length <N>`
- `--max_target_length <N>`
