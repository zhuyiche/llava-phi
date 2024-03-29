<div style="display: flex; align-items: center;">
  <a href="https://arxiv.org/abs/2403.06199">
    <h1>LLaVA-Phi & Mipha: Towards Multimodal Small Language Models</h1>
  </a>
</div>

<div align="center">
<img src="docs/mipha.jpg" width="20%">
</div>

* **LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2402.03766-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2401.02330)



* **Mipha: A Comprehensive Overhaul of Multimodal Assistant with Small Language Models** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2312.16886-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.06199) 


## 📸 Release
* **`March. 23th, 2024`**: Our model 🔥🔥🔥 **Mipha-3B** and corresponding training codes are released.
* **`Jan. 26th, 2024`**:Now you can download our [model weight]((#llava-weights)).
* **`Jan. 15th, 2024`**:Our model and training codes are released.
* **`Jan. 5th, 2024`**: Our codes are currently undergoing an internal review and will be released shortly (expected next week)

## Model Zoo
## Mipha & LLaVA-Phi
| Model | LLM | VQAv2 | GQA | SQA<sup>I</sup> | VQA<sup>T</sup> | POPE | MME<sup>P</sup>  | MMB |
|-------|-------|---|-----|-------|-------|-------|-------|-------|
| <div style="width: 93pt"> LLaVA-Phi-3B  | <div style="width: 91pt"> Phi-2-2.7B | 71.4 | - | 68.4 | 48.6 | 85.0 | 1335.1 | 59.8 |
| <div style="width: 93pt"> Mipha-1.6B   | <div style="width: 91pt"> Phi-1.5-1.3B | 77.5 | 62.7 | 58.3 | 45.6 | **86.9** | 1203.1 | 57.7 |
| <div style="width: 93pt"> Mipha-2.4B   | <div style="width: 91pt"> Gemma-2B | 79.5 | 63.3 | 65.3 | 52.4 | 86.6 | 1397.1 | 59.4 |
| <div style="width: 93pt"> Mipha-3B   | <div style="width: 91pt"> Phi-2-2.7B | **81.3** | **63.9** | **70.9** | **56.6** | 86.7 | **1488.9** | **69.7** | 


## Contents
- [Install](#install)
- [Mipha Weights](#Mipha-weights)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to llava-phi folder
```bash
git clone https://github.com/zhuyiche/Mipha.git
cd Mipha
```

2. Install Package
```Shell
conda create -n mipha python=3.10 -y
conda activate mipha
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Mipha Weights
Download Mipha-3B at [huggingface](https://huggingface.co/zhumj34/Mipha-3B)

## Train

Mipha training consists of two stages: (1) feature alignment stage: use [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; 
(2) visual instruction tuning stage: visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

### Hyperparameters
The hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: | ---: | ---: |
| Mipha          | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: |-------:| ---: | ---: |
| Mipha      | 128 | 2e-5 |      2 | 2048 | 0 |

### Download base checkpoints

Our base model is phi-2. You should download the weights from [here](https://huggingface.co/susnato/phi-2), and change the `--model_name_or_path` in [`get_base_model.sh`](https://github.com/zhuyiche/Mipha/blob/main/scripts/mipha/get_base_model.sh). <br>
Our vision encoder is SigLIP-SO (0.4B). You should download the weights from [here](https://huggingface.co/google/siglip-so400m-patch14-384).

### Integrate the model
Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). <br>

Then, you should integrate phi-2 and SigLIP-SO into a single model by running the following script:
```bash
bash ./script/mipha/get_base_model.sh
```

### Pretrain (feature alignment)


```bash
bash ./scripts/mipha/pretrain.sh
```

### Visual Instruction Tuning

Please refer [here](https://github.com/haotian-liu/LLaVA/blob/9a26bd1435b4ac42c282757f2c16d34226575e96/README.md#visual-instruction-tuning) to prepare the instruction tuning data.

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/zhuyiche/Mipha/blob/main/scripts/mipha/finetune.sh).

```bash
bash ./scripts/mipha/finetune.sh
```

## Evaluation

To ensure the reproducibility, we evaluate the models with greedy decoding. 

See [Evaluation.md](https://github.com/zhuyiche/Mipha/blob/main/docs/Evaluation.md).

## CLI Inference Guide
You can chat about images using Mipha without the Gradio interface. Here is an example command:
```bash
python -m mipha.serve.cli \
    --model-path /path/to/mipha-3B \
    --image-file "mipha/serve/examples/extreme_ironing.jpg" \
    --conv-mode phi
```

## Citation

If you find LLaVA-Phi or Mipha useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:

```
@misc{zhu2024llavaphi,
      title={LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model}, 
      author={Yichen Zhu and Minjie Zhu and Ning Liu and Zhicai Ou and Xiaofeng Mou and Jian Tang},
      year={2024},
      eprint={2401.02330},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zhu2024comprehensive,
  title={A Comprehensive Overhaul of Multimodal Assistant with Small Language Models},
  author={Zhu, Minjie and Zhu, Yichen and Liu, Xin and Liu, Ning and Xu, Zhiyuan and Shen, Chaomin and Peng, Yaxin and Ou, Zhicai and Feng, Feifei and Tang, Jian},
  journal={arXiv preprint arXiv:2403.06199},
  year={2024}
}

```

## Acknowledgement
We build our project based on
- [LLaVA](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for vision language assistant
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use this codebase to finetune SLMs
- [Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf): We use this codebase to instruct-tune SLMs
