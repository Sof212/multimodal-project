import clip
import torch


@torch.no_grad()
def encode_images(model, preprocess, images, device, batch_size=64):
    """
    Compute L2-normalised CLIP embeddings for a list of images.

    Parameters
    ----------
    model : torch.nn.Module
        CLIP model.
    preprocess : callable
        CLIP image transform.
    images : list of PIL.Image.Image
        Images to encode.
    device : str
        Device on which to run the model.
    batch_size : int, default=64
        Number of images encoded at once.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_images, embed_dim)`` on CPU.
    """
    features = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(img) for img in images[start:start + batch_size]])
        emb = model.encode_image(batch.to(device)).float()
        features.append(emb / emb.norm(dim=-1, keepdim=True))
    return torch.cat(features).cpu()


@torch.no_grad()
def encode_texts(model, texts, device, prefix="", batch_size=256):
    """
    Compute L2-normalised CLIP embeddings for a list of texts.

    Parameters
    ----------
    model : torch.nn.Module
        CLIP model.
    texts : list of str
        Texts to encode. Texts longer than the context length are
        truncated.
    device : str
        Device on which to run the model.
    prefix : str, default=""
        String prepended to every text (e.g. ``"This is "``).
    batch_size : int, default=256
        Number of texts encoded at once.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_texts, embed_dim)`` on CPU.
    """
    features = []
    for start in range(0, len(texts), batch_size):
        chunk = [prefix + t for t in texts[start:start + batch_size]]
        tokens = clip.tokenize(chunk, truncate=True).to(device)
        emb = model.encode_text(tokens).float()
        features.append(emb / emb.norm(dim=-1, keepdim=True))
    return torch.cat(features).cpu()


def cosine_similarity_matrix(model, preprocess, images, texts, device, prefix="This is "):
    """
    Compute the text-image cosine similarity matrix.

    Parameters
    ----------
    model : torch.nn.Module
        CLIP model.
    preprocess : callable
        CLIP image transform.
    images : list of PIL.Image.Image
        Images, aligned with ``texts``.
    texts : list of str
        Captions, aligned with ``images``.
    device : str
        Device on which to run the model.
    prefix : str, default="This is "
        Prompt prefix prepended to every caption.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_texts, n_images)``; entry ``[i, j]`` is the
        similarity between text ``i`` and image ``j``.
    """
    image_features = encode_images(model, preprocess, images, device)
    text_features = encode_texts(model, texts, device, prefix=prefix)
    return (text_features @ image_features.T).numpy()
