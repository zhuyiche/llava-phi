"""This file is modified from https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/run_llava.py"""

import argparse
import random
import torch
import os
import json
import shortuuid

from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets

from mipha.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mipha.conversation import conv_templates, SeparatorStyle
from mipha.model.builder import load_pretrained_model
from mipha.utils import disable_torch_init
from mipha.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from mipha.eval.mmmu_data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from mipha.eval.mmmu_eval_utils import parse_multi_choice_response




def call_mipha_engine_df(args, sample, model, tokenizer=None, processor=None, conv_mode="phi"):
    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep2
    keywords = [stop_str]
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
    image = sample['image']
    if image is not None:
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).half().cuda(),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.eos_token_id,  # Pad token
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        if response.endswith(stop_str):
            response = response[:-len(stop_str)]
        response = response.strip()
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor, conv_mode=args.conv_mode)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples


def main(args):
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # set_seed(args.seed)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    call_model_engine = call_mipha_engine_df

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample['image']:
            # print(sample['image'])
            # image = Image.open(sample['image']).convert('RGB')
            sample['image'] = process_images([sample['image'].convert('RGB')], image_processor, model.config)[0].to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, image_processor)

    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--data-path', type=str, default="MMMU/MMMU")  # hf dataset path.
    parser.add_argument('--output-path', type=str, default='mipha_3b_val.json',
                        help='name of saved json')
    parser.add_argument('--config-path', type=str, default="mipha/eval/mipha.yaml")
    parser.add_argument('--split', type=str, default='validation')
    # parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--conv-mode", type=str, default="phi3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    main(args)
