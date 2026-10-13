import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, save_path, show):
    """
    Save and/or display a figure, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalise.
    save_path : str or None
        Output file path. Nothing is saved if None.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure.
    """
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_image_grid(images, titles, n_cols=4, save_path=None, show=True):
    """
    Display images in a grid with their captions as titles.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Images to display.
    titles : list of str
        One title per image.
    n_cols : int, default=4
        Number of columns.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    n_rows = int(np.ceil(len(images) / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return _finish(fig, save_path, show)


def plot_similarity_matrix(similarity, images, texts, title="Cosine similarity",
                           vmin=0.1, vmax=0.3, fontsize=18, colorbar=False,
                           save_path=None, show=True):
    """
    Plot a text-image similarity heatmap with image thumbnails on top.

    Parameters
    ----------
    similarity : numpy.ndarray
        Matrix of shape ``(n_texts, n_images)``.
    images : list of PIL.Image.Image
        Images shown above the columns.
    texts : list of str
        Captions shown next to the rows.
    title : str, default="Cosine similarity"
        Figure title.
    vmin, vmax : float, default=0.1, 0.3
        Colour scale limits.
    fontsize : int, default=18
        Font size of the row labels.
    colorbar : bool, default=False
        Whether to draw a colour bar.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    n_texts, n_images = similarity.shape
    fig, ax = plt.subplots(figsize=(20, 14))
    im = ax.imshow(similarity, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.set_yticks(range(n_texts), texts, fontsize=fontsize)
    ax.set_xticks([])
    for i, image in enumerate(images):
        ax.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(n_images):
        for y in range(n_texts):
            ax.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xlim([-0.5, n_images - 0.5])
    ax.set_ylim([n_texts + 0.5, -2])
    ax.set_title(title, size=20)
    return _finish(fig, save_path, show)


def plot_zero_shot_predictions(images, top_probs, top_labels, classes,
                               n_show=8, save_path=None, show=True):
    """
    Show images next to a bar chart of their top-k predicted classes.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Classified images.
    top_probs : torch.Tensor
        Probabilities of shape ``(n_images, k)``.
    top_labels : torch.Tensor
        Class indices of shape ``(n_images, k)``.
    classes : list of str
        Class names indexed by ``top_labels``.
    n_show : int, default=8
        Number of images to display.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    n_show = min(n_show, len(images))
    fig = plt.figure(figsize=(16, n_show * 2))
    y = np.arange(top_probs.shape[-1])
    for i in range(n_show):
        ax_img = fig.add_subplot(n_show, 2, 2 * i + 1)
        ax_img.imshow(images[i])
        ax_img.axis("off")
        ax_bar = fig.add_subplot(n_show, 2, 2 * i + 2)
        ax_bar.grid()
        ax_bar.barh(y, top_probs[i].numpy())
        ax_bar.invert_yaxis()
        ax_bar.set_axisbelow(True)
        ax_bar.set_yticks(y, [classes[j] for j in top_labels[i].numpy()])
        ax_bar.set_xlabel("Probability")
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    return _finish(fig, save_path, show)


def plot_training_history(history, save_path=None, show=True):
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    history : list of dict
        One dict per epoch with keys ``epoch``, ``train_loss`` and
        ``val_loss`` (as returned by :func:`clip_roco.train.fit`).
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive loss")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)
