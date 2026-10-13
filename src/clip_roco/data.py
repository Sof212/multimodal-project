import os
import random
from functools import partial

import pandas as pd
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path):
    """
    Open an image file and convert it to RGB.

    Parameters
    ----------
    path : str
        Path to the image.

    Returns
    -------
    PIL.Image.Image
        RGB image.
    """
    return Image.open(path).convert("RGB")


def read_pairs_csv(csv_file, root_dir):
    """
    Read an image/caption CSV file into a list of pairs.

    The first column must contain the image file name and the second
    column the caption. If an image is missing from ``root_dir`` the
    function also looks in the matching ``val`` folder, as some ROCO
    lists mix both splits.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    root_dir : str
        Directory containing the images.

    Returns
    -------
    list of tuple of (str, str)
        ``(image_path, caption)`` pairs.
    """
    frame = pd.read_csv(csv_file)
    pairs = []
    for name, caption in zip(frame.iloc[:, 0], frame.iloc[:, 1]):
        path = os.path.join(root_dir, str(name))
        if not os.path.exists(path):
            path = path.replace("train", "val")
        pairs.append((path, str(caption)))
    return pairs


def parse_roco_captions(file_path):
    """
    Parse a ROCO ``captions.txt`` file (``<id>\\t<caption>`` per line).

    Parameters
    ----------
    file_path : str
        Path to the captions file.

    Returns
    -------
    dict
        Mapping ``image_id -> caption``. Malformed lines are skipped.
    """
    captions = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            captions[key] = value.strip()
    return captions


def sample_caption_sentence(text):
    """
    Pick one random sentence from a multi-sentence caption.

    Parameters
    ----------
    text : str
        Full caption.

    Returns
    -------
    str
        A single non-empty sentence, or the original text if it cannot
        be split.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return random.choice(sentences) if sentences else text


def collate_pairs(batch, preprocess, sample_sentence=False):
    """
    Turn a batch of ``(image_path, caption)`` pairs into model inputs.

    Parameters
    ----------
    batch : list of tuple of (str, str)
        Pairs produced by :func:`read_pairs_csv`.
    preprocess : callable
        CLIP image transform.
    sample_sentence : bool, default=False
        If True, keep a single random sentence of each caption.

    Returns
    -------
    images : torch.Tensor
        Tensor of shape ``(batch, 3, H, W)``.
    texts : list of str
        Captions aligned with ``images``.
    """
    images = torch.stack([preprocess(load_image(path)) for path, _ in batch])
    texts = [sample_caption_sentence(t) if sample_sentence else t for _, t in batch]
    return images, texts


def build_dataloader(pairs, preprocess, batch_size=16, shuffle=False,
                     num_workers=4, drop_last=True, sample_sentence=False):
    """
    Build a PyTorch DataLoader over image/caption pairs.

    Parameters
    ----------
    pairs : list of tuple of (str, str)
        ``(image_path, caption)`` pairs.
    preprocess : callable
        CLIP image transform.
    batch_size : int, default=16
        Number of pairs per batch.
    shuffle : bool, default=False
        Whether to shuffle the pairs at every epoch.
    num_workers : int, default=4
        Number of loading processes.
    drop_last : bool, default=True
        Drop the last incomplete batch. Recommended for contrastive
        training so every batch has the same size.
    sample_sentence : bool, default=False
        See :func:`collate_pairs`.

    Returns
    -------
    torch.utils.data.DataLoader
        Iterable yielding ``(images, texts)`` batches.
    """
    collate_fn = partial(collate_pairs, preprocess=preprocess,
                         sample_sentence=sample_sentence)
    return torch.utils.data.DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def load_skimage_samples(descriptions):
    """
    Load the natural example images shipped with scikit-image.

    Parameters
    ----------
    descriptions : dict
        Mapping ``image_name -> caption`` (e.g. ``NATURAL_DESCRIPTIONS``).

    Returns
    -------
    images : list of PIL.Image.Image
        Loaded images.
    texts : list of str
        Captions aligned with ``images``.
    """
    import skimage

    skimage.data.download_all(directory=None)
    images, texts = [], []
    for filename in sorted(os.listdir(skimage.data_dir)):
        name, ext = os.path.splitext(filename)
        if ext not in (".png", ".jpg") or name not in descriptions:
            continue
        images.append(load_image(os.path.join(skimage.data_dir, filename)))
        texts.append(descriptions[name])
    return images, texts


def load_roco_samples(image_dir, captions, limit=None):
    """
    Load ROCO images with their captions.

    Parameters
    ----------
    image_dir : str
        Directory containing the ``<id>.jpg`` images.
    captions : dict
        Mapping ``image_id -> caption`` (e.g. ``MEDICAL_DESCRIPTIONS`` or
        the output of :func:`parse_roco_captions`).
    limit : int, optional
        Maximum number of samples to load.

    Returns
    -------
    images : list of PIL.Image.Image
        Loaded images.
    texts : list of str
        Captions aligned with ``images``.
    """
    keys = list(captions.keys())[:limit]
    images = [load_image(os.path.join(image_dir, f"{k}.jpg")) for k in keys]
    texts = [captions[k] for k in keys]
    return images, texts
