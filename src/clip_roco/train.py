import json
import os
import time

import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from clip_roco.data import build_dataloader, read_pairs_csv
from clip_roco.model import get_device, load_clip
from clip_roco.utils import ensure_dir, set_seed


def contrastive_loss(logits_per_image, logits_per_text):
    """
    Symmetric InfoNCE loss used to train CLIP.

    Parameters
    ----------
    logits_per_image : torch.Tensor
        Image-to-text logits of shape ``(batch, batch)``.
    logits_per_text : torch.Tensor
        Text-to-image logits of shape ``(batch, batch)``.

    Returns
    -------
    torch.Tensor
        Scalar loss: mean of the image and text cross-entropies, where
        the matching pair of each row is on the diagonal.
    """
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return (F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)) / 2


def build_optimizer(model, learning_rate, epochs, weight_decay=0.0):
    """
    Create the Adam optimizer and its cosine learning-rate schedule.

    Parameters
    ----------
    model : torch.nn.Module
        Model to optimise.
    learning_rate : float
        Initial learning rate.
    epochs : int
        Number of epochs, used as the cosine period.
    weight_decay : float, default=0.0
        L2 penalty.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        Adam optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Cosine annealing scheduler, to be stepped once per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, device, log_every=50):
    """
    Run one training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        CLIP model.
    loader : torch.utils.data.DataLoader
        Training batches of ``(images, texts)``.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : str
        Device on which to run the model.
    log_every : int, default=50
        Print the running loss every ``log_every`` steps.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total = 0.0
    for step, (images, texts) in enumerate(tqdm(loader, desc="train", leave=False)):
        images = images.to(device, non_blocking=True)
        tokens = clip.tokenize(texts, truncate=True).to(device)
        optimizer.zero_grad()
        loss = contrastive_loss(*model(images, tokens))
        loss.backward()
        optimizer.step()
        total += loss.item()
        if step % log_every == 0:
            tqdm.write(f"step {step}/{len(loader)} loss {loss.item():.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """
    Compute the mean contrastive loss on a validation set.

    Parameters
    ----------
    model : torch.nn.Module
        CLIP model.
    loader : torch.utils.data.DataLoader
        Validation batches of ``(images, texts)``.
    device : str
        Device on which to run the model.

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    total = 0.0
    for images, texts in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        tokens = clip.tokenize(texts, truncate=True).to(device)
        total += contrastive_loss(*model(images, tokens)).item()
    return total / max(len(loader), 1)


def fit(config):
    """
    Fine-tune CLIP on image/caption pairs.

    The best model (lowest validation loss) is saved to
    ``<output_dir>/best_checkpoint.pth`` and the loss history to
    ``<output_dir>/history.json``.

    Parameters
    ----------
    config : dict
        Training configuration with keys ``train_csv``,
        ``train_image_dir``, ``val_csv``, ``val_image_dir``,
        ``model_name``, ``seed``, ``batch_size``, ``epochs``,
        ``learning_rate``, ``num_workers``, ``output_dir`` and optionally
        ``weight_decay`` and ``sample_sentence``.

    Returns
    -------
    list of dict
        Per-epoch ``epoch``, ``train_loss``, ``val_loss`` and ``lr``.
    """
    set_seed(config["seed"])
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])

    model, preprocess = load_clip(config["model_name"], device=device)
    optimizer, scheduler = build_optimizer(model, float(config["learning_rate"]),
                                           config["epochs"],
                                           float(config.get("weight_decay", 0.0)))

    train_loader = build_dataloader(
        read_pairs_csv(config["train_csv"], config["train_image_dir"]),
        preprocess, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"],
        sample_sentence=config.get("sample_sentence", False),
    )
    val_loader = build_dataloader(
        read_pairs_csv(config["val_csv"], config["val_image_dir"]),
        preprocess, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )

    history = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_checkpoint.pth"))

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "lr": lr})
        elapsed = time.time() - start
        print(f"epoch {epoch}/{config['epochs']} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | lr {lr:.2e} | {elapsed / 60:.1f} min")

        with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return history
