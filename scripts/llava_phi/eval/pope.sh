#!/bin/bash

python -m llava_phi.eval.model_vqa_loader \
    --model-path ./checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /path/to/data/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode phi-2_v0

python llava_phi/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llavaPhi-v0-3b.jsonl