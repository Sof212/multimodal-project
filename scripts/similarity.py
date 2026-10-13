import argparse
import os

from clip_roco.data import load_roco_samples, load_skimage_samples
from clip_roco.metrics import evaluate_similarity
from clip_roco.model import describe_model, get_device, load_clip
from clip_roco.samples import MEDICAL_DESCRIPTIONS, NATURAL_DESCRIPTIONS
from clip_roco.similarity import cosine_similarity_matrix
from clip_roco.utils import ensure_dir
from clip_roco.visualization import plot_similarity_matrix


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Text-image cosine similarity with CLIP.")
    parser.add_argument("--domain", choices=["natural", "medical"], default="natural")
    parser.add_argument("--model", default="ViT-B/16", help="CLIP architecture.")
    parser.add_argument("--checkpoint", help="Optional fine-tuned state_dict.")
    parser.add_argument("--image-dir", default="data/roco/train/radiology/images",
                        help="ROCO image folder (medical domain only).")
    parser.add_argument("--output-dir", default="outputs/figures")
    return parser.parse_args()


def main():
    """
    Compute, print and plot the similarity matrix for the chosen domain.

    Returns
    -------
    None
    """
    args = parse_args()
    device = get_device()
    model, preprocess = load_clip(args.model, device=device, checkpoint=args.checkpoint)
    print(describe_model(model))

    if args.domain == "natural":
        images, texts = load_skimage_samples(NATURAL_DESCRIPTIONS)
    else:
        images, texts = load_roco_samples(args.image_dir, MEDICAL_DESCRIPTIONS)

    similarity = cosine_similarity_matrix(model, preprocess, images, texts, device)
    print(evaluate_similarity(similarity))

    tag = "finetuned" if args.checkpoint else "pretrained"
    name = f"similarity_{args.domain}_{args.model.replace('/', '-')}_{tag}.png"
    plot_similarity_matrix(similarity, images, texts,
                           title=f"Cosine similarity ({args.domain}, {args.model}, {tag})",
                           save_path=os.path.join(ensure_dir(args.output_dir), name),
                           show=False)


if __name__ == "__main__":
    main()
