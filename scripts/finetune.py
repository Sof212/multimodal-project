import argparse
import os

from clip_roco.train import fit
from clip_roco.utils import load_config
from clip_roco.visualization import plot_training_history


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on ROCO.")
    parser.add_argument("--config", default="configs/finetune_roco.yaml",
                        help="Path to the YAML training configuration.")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override the batch size.")
    return parser.parse_args()


def main():
    """
    Load the configuration, train the model and save the loss curves.

    Returns
    -------
    None
    """
    args = parse_args()
    config = load_config(args.config)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size

    history = fit(config)
    plot_training_history(history, save_path=os.path.join(config["output_dir"], "loss.png"),
                          show=False)


if __name__ == "__main__":
    main()
