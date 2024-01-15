#!/bin/bash

python -m llava_phi.eval.model_vqa_loader \
    --model-path checkpoints/llavaPhi-v0-3b-finetune \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llavaPhi-v0-3b.jsonl \
    --temperature 0 \
    --conv-mode phi-2_v0

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llavaPhi-v0-3b

cd eval_tool

python calculation.py --results_dir answers/llavaPhi-v0-3b
