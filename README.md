# <p style="text-align: center;"> NLP Playground </p>

<p style="text-align: center;">
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue"></a>
<br>
</p>


This repository serves as a playground for exploring Natural Language Processing (NLP) tasks and 
experimenting with popular deep learning models, datasets, and pipelines.

## 1. Machine Translation

As a newcomer to the NLP field, my journey began with a classic NLP task: Machine Translation.

### 1.1 Transformer: PyTorch implementation from scratch

To get good understanding of the classic paper ["Attention is All you Need"](https://arxiv.org/abs/1706.03762), 
I followed this [Youtube video](https://www.youtube.com/watch?v=ISNdQcPhsts) and implemented a basic transformer from scratch.
 
### 1.2 Dataset: Opus-100 from Hugging Face.

The [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) dataset on Hugging Face offers a multilingual resource for training machine translation models. 
This English-centric collection covers 100 languages with millions of sentence pairs for training and evaluation.
Since my first language is Chinese, I chose the "en-zh" subset so that I can assess the translation model's performance.

### 1.3 Tokenizer: Training / Fine-tuning

