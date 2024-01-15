#!/bin/bash

python -m llava_phi.eval.model_vqa_loader \
    --model-path checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode phi-2_v0

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llavaPhi-v0-3b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llavaPhi-v0-3b.json
