# LLaVA-Phi: Small Multi-Modal Assistant

**LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model** [[Paper](https://arxiv.org/pdf/2401.02330)] <br>


## Release
[1/15] Our model and training codes are released.

[1/5] Our codes are currently undergoing an internal review and will be released shortly (expected next week)


## Contents
- [Install](#install)
- [LLaVA-Phi Weights](#llava-weights)
- [Demo](#Demo)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/zhuyiche/llava-phi.git
cd llava-phi
```

2. Install Package
```Shell
conda create -n llava_phi python=3.10 -y
conda activate llava_phi
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## LLaVA-Phi Weights
#Todo

## Demo
#Tdo

## Train

*Below is the latest training configuration for LLaVA v1.5. For legacy models, please refer to README of [this](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) version for now. We'll add them in a separate doc later.*

LLaVA training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data (with VQA data from academic-oriented tasks) to teach the model to follow multimodal instructions.

LLaVA is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below. We note that the hyperparameters may not be the same as we reported in the arxiv paper, as this is an on-going project and we are making frequent changes on our codes.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: | ---: | ---: |
| LLaVA-Phi      | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: | ---: | ---: |
| LLaVA-Phi      | 128 | 2e-5 | 1 | 2048 | 0 |

### Download base checkpoints

Our base model phi-2, you should download the weights from [here](https://huggingface.co/susnato/phi-2).

### Intergate the model

### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

### Visual Instruction Tuning

1. Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Start training!

You may download our pretrained projectors in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). It is not recommended to use legacy projectors, as they may be trained with a different version of the codebase, and if any option is off, the model will not function/train as we expected.

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune.sh).

New options to note:

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length True`: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.

## Evaluation

To ensure the reproducibility, we evaluate the models with greedy decoding.

See [Evaluation.md].

### Usage and License Notices
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses. This project is licensed permissively under the Apache 2.0 license and does not impose any additional constraints. 

## Citation

If you find LLaVA-Phi useful for your research and applications, please cite using this BibTeX:
```bibtex

@article{zhu2024llava,
  title={LLaVA-$$\backslash$phi $: Efficient Multi-Modal Assistant with Small Language Model},
  author={Zhu, Yichen and Zhu, Minjie and Liu, Ning and Ou, Zhicai and Mou, Xiaofeng and Tang, Jian},
  journal={arXiv preprint arXiv:2401.02330},
  year={2024}
}
```

## Acknowledgement
We build our project based on
- [LLaVA](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for vision language assistant
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use this codebase to finetune Phi model
