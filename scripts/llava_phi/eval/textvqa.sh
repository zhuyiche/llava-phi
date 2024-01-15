#!/bin/bash

python -m llava_phi.eval.model_vqa_loader \
    --model-path ./checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /path/to/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode phi-2_v0

python -m llava_phi.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llavaPhi-v0-3b.jsonl
