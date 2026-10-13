# CLIP for Medical Images (ROCO)

Evaluating and fine-tuning [OpenAI CLIP](https://github.com/openai/CLIP) on radiology
image–caption pairs from the [ROCO dataset](https://github.com/razorx89/roco-dataset).

General-purpose vision-language models such as CLIP are trained on natural web images.
This project measures how well they transfer to the medical domain and how much a short
contrastive fine-tuning on radiology data closes the gap.

## What this project does

- **Text–image similarity**: computes the cosine similarity between CLIP image and text
  embeddings, on natural images and on radiology images (CT, MRI, X-ray, angiography).
- **Model comparison**: benchmarks every public CLIP architecture (RN50 … ViT-L/14@336px)
  on medical image–caption retrieval.
- **Zero-shot classification**: evaluates CLIP without training on STL-10 / CIFAR-100.
- **Fine-tuning**: contrastive (InfoNCE) fine-tuning of CLIP ViT-B/16 on ROCO, with
  validation-based checkpointing.

## Results

<!-- Fill these tables with YOUR numbers from scripts/compare_models.py and scripts/similarity.py -->

**Medical retrieval (8 ROCO pairs), pretrained CLIP**

| Model | Text→Image Acc (%) | Image→Text Acc (%) | Diag − RowMean |
|-------|-------------------:|-------------------:|---------------:|
| ViT-B/16 | – | – | – |
| ViT-L/14 | – | – | – |

**Before vs after fine-tuning (ViT-B/16)**

| | Text→Image Acc (%) | Image→Text Acc (%) | Diag − RowMean |
|-|-------------------:|-------------------:|---------------:|
| Pretrained | – | – | – |
| Fine-tuned on ROCO | – | – | – |

**Zero-shot STL-10 (2,500 test images)**: ViT-B/32 top-1 accuracy = –

<!-- Add figures saved in outputs/figures/ to a docs/ folder and reference them here, e.g.:
![Similarity matrix](docs/similarity_medical_finetuned.png) -->

## Project structure

```
clip-medical-roco/
├── configs/
│   └── finetune_roco.yaml      # training hyper-parameters and data paths
├── data/
│   └── README.md               # how to download and arrange ROCO (data is git-ignored)
├── notebooks/
│   └── clip_roco_demo.ipynb    # walkthrough using the package (Colab-ready)
├── scripts/
│   ├── similarity.py           # similarity matrix, natural or medical, pretrained or fine-tuned
│   ├── compare_models.py       # benchmark all CLIP architectures
│   ├── zero_shot.py            # zero-shot classification on STL-10 / CIFAR-100
│   └── finetune.py             # fine-tune CLIP on ROCO
├── src/clip_roco/
│   ├── data.py                 # ROCO parsing, image loading, DataLoader
│   ├── model.py                # CLIP loading and description
│   ├── similarity.py           # embeddings and cosine similarity
│   ├── zero_shot.py            # zero-shot prediction and accuracy
│   ├── metrics.py              # retrieval metrics and model comparison
│   ├── train.py                # contrastive loss and training loop
│   ├── visualization.py        # plots
│   ├── samples.py              # example image/caption pairs
│   └── utils.py                # seeding, config, folders
├── tests/                      # unit tests (pytest)
├── outputs/                    # figures, tables, checkpoints (git-ignored)
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<your-username>/clip-medical-roco.git
cd clip-medical-roco
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

A CUDA GPU is strongly recommended for fine-tuning; the evaluation scripts also run on CPU.

## Usage

Prepare the data as described in [`data/README.md`](data/README.md), then:

```bash
# Similarity matrices
python scripts/similarity.py --domain natural
python scripts/similarity.py --domain medical --model ViT-B/16

# Compare all CLIP architectures on medical retrieval
python scripts/compare_models.py
python scripts/compare_models.py --captions-file data/roco/val/radiology/captions.txt \
    --image-dir data/roco/val/radiology/images --limit 15

# Zero-shot classification
python scripts/zero_shot.py --dataset stl10 --model ViT-B/32

# Fine-tuning
python scripts/finetune.py --config configs/finetune_roco.yaml

# Evaluate the fine-tuned model
python scripts/similarity.py --domain medical \
    --checkpoint outputs/checkpoints/vit-b16-roco/best_checkpoint.pth
```

Run the tests with `pytest`.

## Method

CLIP is fine-tuned with the symmetric contrastive objective of the original paper: for a
batch of *N* image–caption pairs, the model predicts which of the *N × N* pairings are
correct, using cross-entropy on the image-to-text and text-to-image logits. Training uses
Adam (lr 3e-6) with cosine annealing; the checkpoint with the lowest validation loss is kept.

Retrieval metrics on a square similarity matrix (captions × images, matching pairs on the
diagonal):

- **Text→Image / Image→Text accuracy**: share of rows / columns whose maximum is on the diagonal.
- **Diag − RowMean**: mean gap between the matching score and the other scores of its row
  (how clearly the model separates the right pair).

## Credits

This project started from the CLIP tutorial notebook of the *Master IA – Vision et Machines
Intelligentes* (Université Paris Cité), created by Camille Kurtz with Xiaoyang Wei (LIPADE).
Extensions in this repository: refactoring into a tested Python package, multi-architecture
benchmark, retrieval metrics, zero-shot evaluation on STL-10, and fixes to the fine-tuning loop.

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
- Pelka et al., *Radiology Objects in COntext (ROCO): A Multimodal Image Dataset*, 2018.
