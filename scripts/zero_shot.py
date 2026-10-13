import argparse
import os
import random

from torchvision import datasets

from clip_roco.model import get_device, load_clip
from clip_roco.utils import ensure_dir, set_seed
from clip_roco.visualization import plot_zero_shot_predictions
from clip_roco.zero_shot import top1_accuracy, zero_shot_predict


def load_dataset(name, root):
    """
    Download a torchvision classification test set.

    Parameters
    ----------
    name : {"stl10", "cifar100"}
        Dataset name.
    root : str
        Download directory.

    Returns
    -------
    torch.utils.data.Dataset
        Dataset returning ``(PIL image, label)`` pairs, with a
        ``classes`` attribute.
    """
    if name == "stl10":
        return datasets.STL10(root=root, split="test", download=True)
    return datasets.CIFAR100(root=root, train=False, download=True)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Zero-shot classification with CLIP.")
    parser.add_argument("--dataset", choices=["stl10", "cifar100"], default="stl10")
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--data-root", default="data/torchvision")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """
    Run zero-shot classification on a random subset and report accuracy.

    Returns
    -------
    None
    """
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    dataset = load_dataset(args.dataset, args.data_root)
    indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))
    images = [dataset[i][0] for i in indices]
    labels = [dataset[i][1] for i in indices]

    model, preprocess = load_clip(args.model, device=device)
    top_probs, top_labels = zero_shot_predict(model, preprocess, images, dataset.classes, device)
    print(f"{args.dataset} | {args.model} | top-1 accuracy: "
          f"{top1_accuracy(top_labels, labels):.2f}% on {len(images)} images")

    name = f"zero_shot_{args.dataset}_{args.model.replace('/', '-')}.png"
    plot_zero_shot_predictions(images, top_probs, top_labels, dataset.classes,
                               save_path=os.path.join(ensure_dir(args.output_dir), name),
                               show=False)


if __name__ == "__main__":
    main()
