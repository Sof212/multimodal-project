import numpy as np
import pandas as pd


def retrieval_accuracy(similarity, axis):
    """
    Compute top-1 retrieval accuracy on a square similarity matrix.

    Parameters
    ----------
    similarity : numpy.ndarray
        Matrix of shape ``(n, n)`` where ``[i, j]`` compares text ``i``
        with image ``j`` and the diagonal holds the matching pairs.
    axis : int
        ``1`` for text-to-image retrieval (best image per text),
        ``0`` for image-to-text retrieval (best text per image).

    Returns
    -------
    float
        Accuracy between 0 and 100.
    """
    predictions = np.argmax(similarity, axis=axis)
    return float(np.mean(predictions == np.arange(similarity.shape[0])) * 100)


def diagonal_margin(similarity):
    """
    Mean gap between each matching score and the other scores of its row.

    Parameters
    ----------
    similarity : numpy.ndarray
        Square similarity matrix.

    Returns
    -------
    float
        Average of ``sim[i, i] - mean(sim[i, j != i])``. Higher means the
        model separates matching pairs more clearly.
    """
    n = similarity.shape[0]
    diag = np.diag(similarity)
    off_diag_mean = (similarity.sum(axis=1) - diag) / (n - 1)
    return float(np.mean(diag - off_diag_mean))


def evaluate_similarity(similarity):
    """
    Compute all retrieval metrics for one similarity matrix.

    Parameters
    ----------
    similarity : numpy.ndarray
        Square similarity matrix.

    Returns
    -------
    dict
        Text-to-image accuracy, image-to-text accuracy, their average,
        mean diagonal score and diagonal margin.
    """
    t2i = retrieval_accuracy(similarity, axis=1)
    i2t = retrieval_accuracy(similarity, axis=0)
    return {
        "Text→Image Acc (%)": round(t2i, 2),
        "Image→Text Acc (%)": round(i2t, 2),
        "Average Acc (%)": round((t2i + i2t) / 2, 2),
        "Diagonal Mean": round(float(np.mean(np.diag(similarity))), 4),
        "Diag Minus RowMean": round(diagonal_margin(similarity), 4),
    }


def compare_models(similarities):
    """
    Build a comparison table from several similarity matrices.

    Parameters
    ----------
    similarities : dict
        Mapping ``model_name -> similarity matrix``.

    Returns
    -------
    pandas.DataFrame
        One row per model, sorted by average accuracy (descending).
    """
    rows = [{"Model": name, **evaluate_similarity(sim)} for name, sim in similarities.items()]
    return pd.DataFrame(rows).sort_values("Average Acc (%)", ascending=False).reset_index(drop=True)
