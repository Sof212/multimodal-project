import argparse
import os

from clip_roco.data import load_roco_samples, parse_roco_captions
from clip_roco.metrics import compare_models
from clip_roco.model import get_device, list_models, load_clip
from clip_roco.samples import MEDICAL_DESCRIPTIONS
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
    parser = argparse.ArgumentParser(description="Compare CLIP architectures on ROCO retrieval.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="CLIP architectures to compare (default: all).")
    parser.add_argument("--image-dir", default="data/roco/train/radiology/images")
    parser.add_argument("--captions-file", default=None,
                        help="ROCO captions.txt; if omitted, the 8 built-in samples are used.")
    parser.add_argument("--limit", type=int, default=15,
                        help="Number of pairs taken from --captions-file.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    """
    Evaluate every model on the same pairs and save a comparison table.

    Returns
    -------
    None
    """
    args = parse_args()
    device = get_device()
    models = args.models or list_models()

    if args.captions_file:
        captions = parse_roco_captions(args.captions_file)
        images, texts = load_roco_samples(args.image_dir, captions, limit=args.limit)
    else:
        images, texts = load_roco_samples(args.image_dir, MEDICAL_DESCRIPTIONS)

    figures_dir = ensure_dir(os.path.join(args.output_dir, "figures"))
    similarities = {}
    for name in models:
        print(f"Evaluating {name}...")
        model, preprocess = load_clip(name, device=device)
        similarities[name] = cosine_similarity_matrix(model, preprocess, images, texts, device)
        if not args.no_plots:
            plot_similarity_matrix(similarities[name], images, texts,
                                   title=f"Cosine similarity - {name}", colorbar=True,
                                   save_path=os.path.join(figures_dir,
                                                          f"compare_{name.replace('/', '-')}.png"),
                                   show=False)
        del model

    table = compare_models(similarities)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(args.output_dir, "model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
