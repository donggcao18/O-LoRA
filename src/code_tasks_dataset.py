# coding=utf-8
"""Dataset adapter for single code-related seq2seq tasks.

This module integrates external HuggingFace datasets into the training pipeline used by `run_uie_lora.py`.

It produces a `datasets.DatasetDict` with splits: train/validation/test.
Each example is already tokenized and contains:
- input_ids
- attention_mask
- labels
- task (string)

This keeps the rest of the pipeline (Trainer, model, LoRA) unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import datasets
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from src.task_info import INSTRUCTION_POOL, INSTRUCTION_SPLIT_POLICY, TASK_SPECS as TASK_SPECS_META


SUPPORTED_TASKS = [
    "CodeTrans",
    "CodeSearchNet",
    "BFP",
    "CONCODE",
    "TheVault_Csharp",
    "KodCode",
    "RunBugRun",
    "CoST",
]

# Per-task defaults (taken from the user's original dataloader).
DEFAULT_MAX_SOURCE_LENGTH = {
    "CodeTrans": 320,
    "CodeSearchNet": 256,
    "BFP": 130,
    "CONCODE": 320,
    "TheVault_Csharp": 256,
    "KodCode": 256,
    "RunBugRun": 256,
}

DEFAULT_MAX_TARGET_LENGTH = {
    "CodeTrans": 256,
    "CodeSearchNet": 128,
    "BFP": 120,
    "CONCODE": 150,
    "TheVault_Csharp": 128,
    "KodCode": 256,
    "RunBugRun": 256,
}


@dataclass
class CodeTaskSpec:
    dataset_name: str
    text_key: str
    label_key: str
    # If dataset doesn't provide validation/test, we will split from train.
    is_train_only: bool = False
    # Some datasets require extra kwargs
    dataset_kwargs: Optional[dict] = None
    # Optional filter function for train split
    filter_language: Optional[str] = None


TASK_SPECS: Dict[str, CodeTaskSpec] = {
    "CONCODE": CodeTaskSpec(
        dataset_name="AhmedSSoliman/CodeXGLUE-CONCODE",
        text_key="nl",
        label_key="code",
        dataset_kwargs=None,
    ),
    "CodeTrans": CodeTaskSpec(
        dataset_name="CM/codexglue_codetrans",
        text_key="java",
        label_key="cs",
        dataset_kwargs=None,
    ),
    "CodeSearchNet": CodeTaskSpec(
        dataset_name="semeru/code-text-ruby",
        text_key="code",
        label_key="docstring",
        dataset_kwargs=None,
    ),
    "BFP": CodeTaskSpec(
        dataset_name="ayeshgk/code_x_glue_cc_code_refinement_annotated",
        text_key="buggy",
        label_key="fixed",
        dataset_kwargs=None,
    ),
    "TheVault_Csharp": CodeTaskSpec(
        dataset_name="Fsoft-AIC/the-vault-function",
        text_key="code",
        label_key="docstring",
        dataset_kwargs={"languages": ["c_sharp"]},
    ),
    "KodCode": CodeTaskSpec(
        dataset_name="KodCode/KodCode-V1-SFT-R1",
        text_key="question",
        label_key="solution",
        is_train_only=True,
        dataset_kwargs=None,
    ),
    "RunBugRun": CodeTaskSpec(
        dataset_name="ASSERT-KTH/RunBugRun-Final",
        text_key="buggy_code",
        label_key="fixed_code",
        is_train_only=True,
        filter_language="ruby",
        dataset_kwargs=None,
    ),
    "CoST": CodeTaskSpec(
        dataset_name="dongg18/CoST",
        text_key="lang1",
        label_key="lang2",
        dataset_kwargs=None,
    ),
}


def _to_string(value) -> str:
    if value is None:
        return ""
    return str(value)


def _split_train_only(dataset: Dataset, split: str, *, val_size: int, test_size: int, seed: int = 42) -> Dataset:
    """Create deterministic train/validation/test from a train-only dataset."""
    tmp = dataset.train_test_split(test_size=test_size, seed=seed)
    test_ds = tmp["test"]

    tmp2 = tmp["train"].train_test_split(test_size=val_size, seed=seed)
    train_ds = tmp2["train"]
    val_ds = tmp2["test"]

    mapping = {"train": train_ds, "validation": val_ds, "test": test_ds}
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'. Expected one of {list(mapping.keys())}")
    return mapping[split]


def _select_subset(ds: Dataset, k: int, seed: int) -> Dataset:
    if k <= 0:
        return ds
    rng = np.random.default_rng(seed)
    num_samples = min(k, len(ds))
    idx = rng.choice(np.arange(len(ds)), size=num_samples, replace=False)
    return ds.select(idx.tolist())


def _render_instruction(task: str, raw_input: str) -> str:
    """Minimal instruction renderer.

    For now keep it simple and stable. You can replace with your `task_info.py` pools later.
    """
    return f"Task: {task}\n{raw_input}" + " </s>"


def _get_candidate_instruction_pool(task: str, split_name: str):
    meta = TASK_SPECS_META.get(task)
    if not meta:
        raise ValueError(f"No TASK_SPECS meta found for task '{task}'")
    task_type = meta.get("task_type")
    pool = INSTRUCTION_POOL.get(task_type, [])
    if not pool:
        raise ValueError(f"No instruction templates defined for task_type '{task_type}'")

    policy = INSTRUCTION_SPLIT_POLICY.get(split_name, INSTRUCTION_SPLIT_POLICY["train"])
    scope = policy.get("pool_scope")
    if scope == "full":
        return pool
    if scope == "head_fraction":
        fraction = float(policy.get("fraction", 0.75))
        if fraction <= 0:
            raise ValueError(f"Invalid fraction {fraction} for split '{split_name}'")
        head_size = max(1, int(len(pool) * fraction))
        return pool[:head_size]

    raise ValueError(f"Unknown pool_scope '{scope}' for split '{split_name}'")


def _select_instruction_template(task: str, sample_key: str, split_name: str, split_seed: int) -> str:
    candidate_pool = _get_candidate_instruction_pool(task, split_name)
    random_key = f"{split_seed}::{split_name}::{sample_key}"
    idx = int(hashlib.md5(random_key.encode("utf-8")).hexdigest(), 16) % len(candidate_pool)
    return candidate_pool[idx]


def _render_pooled_instruction(task: str, raw_input: str, sample_key: str, split_name: str, split_seed: int) -> str:
    meta = TASK_SPECS_META[task]
    template = _select_instruction_template(task, sample_key, split_name, split_seed)

    format_values: Dict[str, str] = {
        "language": meta.get("language", "code"),
        "description": raw_input,
        "code": raw_input,
        "source_lang": meta.get("source_lang", meta.get("language", "source language")),
        "target_lang": meta.get("target_lang", "target language"),
    }
    return template.format(**format_values)


def build_code_task_dataset(
    *,
    tokenizer,
    task: str,
    split_seed: int = 42,
    shuffle_seed: int = 0,
    k: int = -1,
    max_target_length: int = 128,
    max_source_length: int = 512,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """Build a tokenized DatasetDict for a single task."""
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task '{task}'. Supported: {sorted(TASK_SPECS.keys())}")

    # Apply per-task default lengths if caller uses the generic defaults.
    if max_source_length == 512 and task in DEFAULT_MAX_SOURCE_LENGTH:
        max_source_length = DEFAULT_MAX_SOURCE_LENGTH[task]
    if max_target_length == 128 and task in DEFAULT_MAX_TARGET_LENGTH:
        max_target_length = DEFAULT_MAX_TARGET_LENGTH[task]

    spec = TASK_SPECS[task]
    dataset_kwargs = dict(spec.dataset_kwargs or {})

    def load_split(split_name: str) -> Dataset:
        if task == "TheVault_Csharp":
            if split_name == "train":
                return load_dataset(spec.dataset_name, cache_dir=cache_dir, split_set="train/small", **dataset_kwargs)
            return load_dataset(spec.dataset_name, cache_dir=cache_dir, split_set=split_name, **dataset_kwargs)

        # Most datasets accept split=...
        return load_dataset(spec.dataset_name, cache_dir=cache_dir, split=split_name, **dataset_kwargs)

    # Load raw splits
    if spec.is_train_only:
        full_train = load_split("train")
        if spec.filter_language:
            # dataset.filter() is deterministic
            full_train = full_train.filter(lambda e: e.get("language") == spec.filter_language)

        train_ds = _split_train_only(full_train, "train", val_size=5000, test_size=5000, seed=split_seed)
        val_ds = _split_train_only(full_train, "validation", val_size=5000, test_size=5000, seed=split_seed)
        test_ds = _split_train_only(full_train, "test", val_size=5000, test_size=5000, seed=split_seed)
    else:
        # If dataset doesn't have a split, this will raise and user can adjust.
        train_ds = load_split("train")
        val_ds = load_split("validation")
        test_ds = load_split("test")

    # Shuffle / subset
    train_ds = train_ds.shuffle(seed=shuffle_seed)
    val_ds = val_ds.shuffle(seed=shuffle_seed)
    test_ds = test_ds.shuffle(seed=shuffle_seed)

    if k != -1:
        train_ds = _select_subset(train_ds, k=k, seed=shuffle_seed)
        val_ds = _select_subset(val_ds, k=min(k, len(val_ds)), seed=shuffle_seed + 1)
        test_ds = _select_subset(test_ds, k=min(k, len(test_ds)), seed=shuffle_seed + 2)

    text_key = spec.text_key
    label_key = spec.label_key

    def preprocess(example, *, split_name: str):
        raw_input = _to_string(example.get(text_key))
        raw_label = _to_string(example.get(label_key))

        sample_uid = hashlib.md5((task + "||" + raw_input).encode("utf-8")).hexdigest()

        instruction = _render_pooled_instruction(
            task=task,
            raw_input=raw_input,
            sample_key=f"{task}::{sample_uid}",
            split_name=split_name,
            split_seed=split_seed,
        )

        source_text = instruction + " </s>"
        target_text = raw_label + " </s>"

        model_inputs = tokenizer(
            source_text,
            padding="max_length",
            truncation=True,
            max_length=max_source_length,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_text,
                padding="max_length",
                truncation=True,
                max_length=max_target_length,
            )["input_ids"]

        # Replace pad token id's in the labels by -100 to ignore loss on padding.
        pad_token_id = tokenizer.pad_token_id
        labels = [(lid if lid != pad_token_id else -100) for lid in labels]

        model_inputs["labels"] = labels
        return model_inputs

    train_ds = train_ds.map(lambda e: preprocess(e, split_name="train"), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda e: preprocess(e, split_name="validation"), remove_columns=val_ds.column_names)
    test_ds = test_ds.map(lambda e: preprocess(e, split_name="test"), remove_columns=test_ds.column_names)

    return DatasetDict(train=train_ds, validation=val_ds, test=test_ds)
