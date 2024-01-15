#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava_phi.eval.model_vqa_mmbench \
    --model-path checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llavaPhi-v0-3b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi-2_v0

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llavaPhi-v0-3b