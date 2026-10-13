# Data

Datasets are **not** versioned in this repository.

## ROCO (Radiology Objects in COntext)

Download ROCO from the official repository: https://github.com/razorx89/roco-dataset
(follow its instructions to fetch the images), then arrange it as follows:

```
data/
└── roco/
    ├── roco-list.csv            # training pairs: <image file name>,<caption>
    ├── local-val-list.csv       # validation pairs: <image file name>,<caption>
    ├── train/radiology/
    │   ├── images/ROCO_00002.jpg ...
    │   └── captions.txt         # <id>\t<caption>
    └── val/radiology/
        ├── images/ ...
        └── captions.txt
```

The CSV files must have a header row; the first column is the image file name
and the second column is the caption.

Paths can be changed in `configs/finetune_roco.yaml` or with the `--image-dir`
/ `--captions-file` options of the scripts.

## Torchvision datasets

STL-10 and CIFAR-100 are downloaded automatically by `scripts/zero_shot.py`
into `data/torchvision/`.
