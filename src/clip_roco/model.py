import clip
import numpy as np
import torch


def get_device():
    """
    Return the best available device.

    Returns
    -------
    str
        ``"cuda"`` if a GPU is available, otherwise ``"cpu"``.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_models():
    """
    List the pretrained CLIP architectures provided by OpenAI.

    Returns
    -------
    list of str
        Model names accepted by :func:`load_clip`.
    """
    return clip.available_models()


def load_clip(model_name="ViT-B/16", device=None, checkpoint=None):
    """
    Load a CLIP model and its image preprocessing transform.

    Parameters
    ----------
    model_name : str, default="ViT-B/16"
        Name of the CLIP architecture (see :func:`list_models`).
    device : str, optional
        Target device. Defaults to :func:`get_device`.
    checkpoint : str, optional
        Path to a fine-tuned ``state_dict`` to load on top of the
        pretrained weights.

    Returns
    -------
    model : torch.nn.Module
        CLIP model in float32 and evaluation mode.
    preprocess : callable
        Torchvision transform expected by the model.
    """
    device = device or get_device()
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.float()
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model, preprocess


def describe_model(model):
    """
    Summarise the main characteristics of a CLIP model.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded CLIP model.

    Returns
    -------
    dict
        Number of parameters, input resolution, context length and
        vocabulary size.
    """
    n_params = int(np.sum([int(np.prod(p.shape)) for p in model.parameters()]))
    return {
        "parameters": n_params,
        "input_resolution": model.visual.input_resolution,
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
    }
