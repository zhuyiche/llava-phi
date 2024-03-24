#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m mipha.eval.model_vqa_science \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v0

python mipha/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$model_name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$model_name-result.json

