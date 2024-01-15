#!/bin/bash

python -m llava_phi.eval.model_vqa_science \
    --model-path ./checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi-2_v0

python llava_phi/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llavaPhi-v0-3b_result.json

