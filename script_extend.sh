#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

export CUDA_VISIBLE_DEVICES=0

python src/run_uie_lora.py  \
    --model_name_or_path Salesforce/codet5p-770m  \
    --cache_dir ./.cache  \
    --dataset_mode code  \
    --code_task CodeTrans  \
    --output_dir logs_and_outputs/code_t5_extended_task \
    --overwrite_output_dir  \
    --do_train \
    --do_eval \
    --do_predict  \
    --predict_with_generate \
    --max_source_length 512  \
    --max_target_length 128  \
    --generation_max_length 128  \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 16  \
    --gradient_accumulation_steps 1  \
    --learning_rate 2e-4  \
    --num_train_epochs 1  \
    --logging_strategy steps \
    --logging_steps 25  \
    --evaluation_strategy steps \
    --eval_steps 50  \
    --save_strategy steps \
    --save_steps 50  \
    --save_total_limit 2  \
    --fp16 \
    --report_to none