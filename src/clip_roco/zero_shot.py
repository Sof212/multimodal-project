import torch

from clip_roco.similarity import encode_images, encode_texts


def build_class_prompts(classes, template="This is a photo of a {}"):
    """
    Turn class names into natural-language prompts.

    Parameters
    ----------
    classes : list of str
        Class names.
    template : str, default="This is a photo of a {}"
        Format string with one ``{}`` placeholder.

    Returns
    -------
    list of str
        One prompt per class.
    """
    return [template.format(c) for c in classes]


def zero_shot_predict(model, preprocess, images, classes, device,
                      template="This is a photo of a {}", top_k=5):
    """
    Classify images without training by comparing them to class prompts.

    The cosine similarities (scaled by 100) are used as softmax logits.

    Parameters
    ----------
    model : torch.nn.Module
        CLIP model.
    preprocess : callable
        CLIP image transform.
    images : list of PIL.Image.Image
        Images to classify.
    classes : list of str
        Candidate class names.
    device : str
        Device on which to run the model.
    template : str, default="This is a photo of a {}"
        Prompt template, see :func:`build_class_prompts`.
    top_k : int, default=5
        Number of predictions returned per image.

    Returns
    -------
    top_probs : torch.Tensor
        Tensor of shape ``(n_images, top_k)`` with class probabilities.
    top_labels : torch.Tensor
        Tensor of shape ``(n_images, top_k)`` with class indices.
    """
    image_features = encode_images(model, preprocess, images, device)
    text_features = encode_texts(model, build_class_prompts(classes, template), device)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return probs.topk(min(top_k, len(classes)), dim=-1)


def top1_accuracy(top_labels, labels):
    """
    Compute top-1 accuracy in percent.

    Parameters
    ----------
    top_labels : torch.Tensor
        Predicted class indices of shape ``(n, k)``; column 0 is the
        top-1 prediction.
    labels : sequence of int
        Ground-truth class indices.

    Returns
    -------
    float
        Accuracy between 0 and 100.
    """
    labels = torch.as_tensor(labels)
    return (top_labels[:, 0] == labels).float().mean().item() * 100
