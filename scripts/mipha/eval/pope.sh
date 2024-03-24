#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /path/to/data/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

python mipha/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$model_name.jsonl