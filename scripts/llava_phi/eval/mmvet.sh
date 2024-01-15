#!/bin/bash

python -m llava_phi.eval.model_vqa \
    --model-path checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode phi-2_v0

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llavaPhi-v0-3b.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llavaPhi-v0-3b.json

