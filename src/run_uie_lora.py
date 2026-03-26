#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys

# IMPORTANT:
# The repo contains a vendored folder `src/peft/` which can shadow the pip package `peft`.
# Transformers >= 4.28 expects newer `peft` symbols (e.g. PeftMixedModel).
# We still need the repo's own modules (e.g. uie_collator) to be importable.
# So: add project root to sys.path, and remove `src/` itself from sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                      # repo root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add

from src.uie_collator import DataCollatorForUIE
from src.uie_dataset_lora import gen_cache_path

from src.uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from src.compute_metrics import compute_metrics, compute_grouped_metrics
from src.model.llama import LlamaForCausalLM_with_lossmask

# Optional: code task dataset adapter (single-task per run)
from src.code_tasks_dataset import build_code_task_dataset
from transformers import DataCollatorForSeq2Seq

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})

    # Dataset mode: 'uie' (repo default) or 'code' (single code task from HF datasets).
    dataset_mode: str = field(
        default="uie",
        metadata={"help": "Dataset mode: 'uie' uses uie_dataset_lora.py; 'code' uses code_tasks_dataset.py"},
    )
    code_task: Optional[str] = field(
        default=None,
        metadata={"help": "When dataset_mode='code', the task name (e.g., CodeTrans, CONCODE, BFP, CoST, ...)"},
    )
    code_k: int = field(
        default=-1,
        metadata={"help": "When dataset_mode='code', optional subset size per split; -1 means use full split."},
    )
    code_shuffle_seed: int = field(
        default=0,
        metadata={"help": "When dataset_mode='code', shuffle seed."},
    )

    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    lamda_1: float = field(default = 0.5)
    lamda_2: float = field(default = 0)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Ensure output/cache directories exist (required for FileLock on Windows).
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Prefer a short, stable cache directory on Windows to avoid deep path/lock issues.
    # If user doesn't pass `--cache_dir`, fall back to a short path.
    hf_cache_dir = model_args.cache_dir or os.path.join(os.path.splitdrive(training_args.output_dir)[0] + os.sep, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)

    # Get datasets
    if data_args.dataset_mode.lower() == "code":
        if not data_args.code_task:
            raise ValueError("--code_task is required when --dataset_mode=code")

        # Delay building tokenized datasets until after tokenizer is loaded.
        raw_datasets = None
    else:
        data_cache_dir = gen_cache_path(training_args.output_dir, data_args)
        os.makedirs(data_cache_dir, exist_ok=True)

        # Validate dataset inputs early (avoid failing deep inside the dataset script).
        if not data_args.data_dir or not os.path.exists(data_args.data_dir):
            raise ValueError(f"Invalid --data_dir: {data_args.data_dir!r}")
        if not data_args.task_config_dir or not os.path.exists(data_args.task_config_dir):
            raise ValueError(f"Invalid --task_config_dir: {data_args.task_config_dir!r}")
        if not data_args.instruction_file or not os.path.exists(data_args.instruction_file):
            raise ValueError(f"Invalid --instruction_file: {data_args.instruction_file!r}")

        # Get the UIE dataset
        # NOTE: On Windows, `datasets` cannot use SIGALRM for timeouts. Also, dataset scripts are treated as custom code.
        raw_datasets = load_dataset(
            os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
            name="default",
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
            download_config=datasets.DownloadConfig(use_etag=False, num_proc=1, max_retries=10, disable_tqdm=False),
            data_dir=data_args.data_dir,
            instruction_file=data_args.instruction_file,
            instruction_strategy=data_args.instruction_strategy,
            task_config_dir=data_args.task_config_dir,
            num_examples=data_args.num_examples,
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        )
        raw_datasets.cleanup_cache_files()

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'adapter' in model_args.model_name_or_path: # load lora-config
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer = transformers.LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
            config.bos_token_id = 1
            config.eos_token_id = 2
            config.pad_token_id = 1
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.pad_token_id = 1
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
            use_fast = model_args.use_fast_tokenizer,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
    else: # load original config
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # If using code dataset mode, build tokenized datasets now that tokenizer is available.
    if data_args.dataset_mode.lower() == "code":
        raw_datasets = build_code_task_dataset(
            tokenizer=tokenizer,
            task=data_args.code_task,
            split_seed=42,
            shuffle_seed=data_args.code_shuffle_seed,
            k=data_args.code_k,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            cache_dir=hf_cache_dir,
        )

    if 'llama' in model_args.model_name_or_path.lower():  # add llama
        model_class = LlamaForCausalLM_with_lossmask
        tokenizer.padding_side = 'left'
    else: 
        model_class = AutoModelForSeq2SeqLM

    if 'adapter' in model_args.model_name_or_path: # add lora-adapter to the original model
        model = model_class.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1
        
    # fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
    # fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
    # optional: lora_A/B is trainable but should not move too far from lorapre_A/B
    # (constrained in "training_step"[uie_trainer_lora.py])
    # NOTE: This custom freezing logic is for the original O-LoRA training scheme (loranew_ only).
    # For code-mode (generic PEFT LoRA), we keep PEFT's default trainable parameters.
    if data_args.dataset_mode.lower() != "code":
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False
            # this module should always be frozen because we change the vocabulary
            elif name.find("shared") != -1:
                param.requires_grad = False

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if data_args.dataset_mode.lower() == "code":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator = DataCollatorForUIE(
            tokenizer,
            model=model,
            padding="longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_dataset_name=data_args.add_dataset_name,
            num_examples=data_args.num_examples,
            input_record_file=data_args.input_record_file
        )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    def compute_code_metrics(dataset, preds, save_prefix=None):
        import collections
        import math
        import re
        import string

        from smooth_bleu_utils import compute_smooth_bleu

        def normalize_text(s: str) -> str:
            # Match the user's BLEU normalization
            def remove_articles(text: str) -> str:
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text: str) -> str:
                return " ".join(text.split())

            def remove_punc(text: str) -> str:
                return "".join(ch for ch in text if ch not in set(string.punctuation))

            s = s.lower().replace("<pad>", "").replace("</s>", "")
            return white_space_fix(remove_articles(remove_punc(s)))

        def _get_ngrams(toks, max_order: int):
            c = collections.Counter()
            for o in range(1, max_order + 1):
                for i in range(0, len(toks) - o + 1):
                    c[tuple(toks[i: i + o])] += 1
            return c

        def compute_bleu(refs, hyps, max_order=4, smooth=False):
            matches = [0] * max_order
            possibles = [0] * max_order
            ref_len = 0
            hyp_len = 0
            for rlist, hyp in zip(refs, hyps):
                r_tokens_list = [r.split() for r in rlist]
                h = hyp.split()
                ref_len += min(len(r) for r in r_tokens_list) if r_tokens_list else 0
                hyp_len += len(h)
                merged = collections.Counter()
                for r in r_tokens_list:
                    merged |= _get_ngrams(r, max_order)
                h_counts = _get_ngrams(h, max_order)
                overlap = h_counts & merged
                for ng in overlap:
                    matches[len(ng) - 1] += overlap[ng]
                for o in range(1, max_order + 1):
                    p = len(h) - o + 1
                    if p > 0:
                        possibles[o - 1] += p
            prec = [0] * max_order
            for i in range(max_order):
                if smooth:
                    prec[i] = (matches[i] + 1.0) / (possibles[i] + 1.0)
                else:
                    prec[i] = (matches[i] / possibles[i]) if possibles[i] > 0 else 0.0
            geo = math.exp(sum((1.0 / max_order) * math.log(p) for p in prec)) if min(prec) > 0 else 0.0
            ratio = float(hyp_len) / max(1, ref_len)
            bp = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / max(ratio, 1e-9))
            return geo * bp

        decoded_preds = skip_instructions(model, preds, tokenizer)

        # For code dataset adapter we store raw label as a plain string in column `labels_text` if present;
        # fallback to `Instance/label` when evaluating on UIE-style datasets.
        if "labels_text" in dataset.column_names:
            references_raw = list(dataset["labels_text"])
            tasks = list(dataset["task"]) if "task" in dataset.column_names else ["unknown"] * len(references_raw)
        else:
            references_raw = [e["Instance"]["label"] for e in dataset]
            tasks = list(dataset["Task"]) if "Task" in dataset.column_names else ["unknown"] * len(references_raw)

        # Corpus BLEU style expects list of refs per example
        refs_norm = [[normalize_text(r)] for r in references_raw]
        hyps_norm = [normalize_text(p) for p in decoded_preds]

        # Split by task: CodeSearchNet + TheVault_Csharp use smooth BLEU; others use classic BLEU.
        smooth_tasks = {"CodeSearchNet", "TheVault_Csharp"}

        scores_by_task = {}
        for t in set(tasks):
            idxs = [i for i, tt in enumerate(tasks) if tt == t]
            if not idxs:
                continue
            t_refs = [refs_norm[i] for i in idxs]
            t_hyps = [hyps_norm[i] for i in idxs]
            if t in smooth_tasks:
                bleu = compute_smooth_bleu(t_refs, t_hyps, n=4, smooth=1, eff_ref_len="shortest",
                                           preserve_case=False, nonorm=False)
                scores_by_task[f"smooth_bleu_for_{t}"] = round(100.0 * bleu, 4)
            else:
                bleu = compute_bleu(t_refs, t_hyps, max_order=4, smooth=False)
                scores_by_task[f"bleu_for_{t}"] = round(100.0 * bleu, 4)

        # Report only per-task scores (no aggregated overall BLEU).
        result = dict(scores_by_task)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = round(float(np.mean(prediction_lens)), 4)

        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for i, pred in enumerate(decoded_preds):
                    fout.write(json.dumps({
                        "task": tasks[i],
                        "label": references_raw[i],
                        "prediction": pred,
                    }) + "\n")

        return result

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required in some PEFT + gradient checkpointing setups when the base model is frozen.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # DEBUG: verify trainable params before training
    if training_args.do_train:
        trainable = []
        all_count = 0
        trainable_count = 0
        for n, p in model.named_parameters():
            all_count += p.numel()
            if p.requires_grad:
                trainable.append((n, tuple(p.shape)))
                trainable_count += p.numel()
        print("=== TRAINABLE PARAMS ===")
        print("num trainable tensors:", len(trainable))
        print("trainable params:", trainable_count)
        print("all params:", all_count)
        print("ratio:", trainable_count / all_count if all_count else 0.0)
        for n, s in trainable[:50]:
            print(n, s)

    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_code_metrics if data_args.dataset_mode.lower() == "code" else compute_rouge_metrics,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        peft_model_id = training_args.output_dir + "/adapter"
        trainer.model.save_pretrained(peft_model_id)  
        tokenizer.save_pretrained(peft_model_id)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logger.info("*** Loading CheckPoint ***")

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    return results


if __name__ == "__main__":
    main()
