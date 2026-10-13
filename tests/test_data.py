import numpy as np
import torch
from PIL import Image

from clip_roco.data import collate_pairs, parse_roco_captions, read_pairs_csv, sample_caption_sentence


def test_parse_roco_captions(tmp_path):
    """
    Tab-separated lines are parsed, malformed lines are skipped.
    """
    path = tmp_path / "captions.txt"
    path.write_text("ROCO_1\tA CT scan.\n\nmalformed line\nROCO_2\t MRI of the knee \n",
                    encoding="utf-8")
    assert parse_roco_captions(path) == {"ROCO_1": "A CT scan.", "ROCO_2": "MRI of the knee"}


def test_sample_caption_sentence():
    """
    A non-empty sentence of the caption is returned.
    """
    text = "First sentence. Second sentence."
    assert sample_caption_sentence(text) in {"First sentence", "Second sentence"}
    assert sample_caption_sentence("...") == "..."


def test_read_pairs_and_collate(tmp_path):
    """
    CSV pairs are read and collated into a tensor batch.
    """
    for name in ["a.jpg", "b.jpg"]:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / name)
    csv = tmp_path / "pairs.csv"
    csv.write_text("name,caption\na.jpg,first\nb.jpg,second\n", encoding="utf-8")

    pairs = read_pairs_csv(csv, str(tmp_path))
    assert [t for _, t in pairs] == ["first", "second"]

    preprocess = lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    images, texts = collate_pairs(pairs, preprocess)
    assert images.shape == (2, 3, 8, 8)
    assert texts == ["first", "second"]
