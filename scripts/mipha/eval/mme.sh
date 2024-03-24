#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $model_name

cd eval_tool

python calculation.py --results_dir answers/$model_name
