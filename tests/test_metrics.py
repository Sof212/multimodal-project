import numpy as np

from clip_roco.metrics import compare_models, diagonal_margin, evaluate_similarity, retrieval_accuracy


def test_perfect_retrieval():
    """
    An identity-like matrix gives 100% accuracy in both directions.
    """
    sim = np.eye(4) * 0.3 + 0.1
    assert retrieval_accuracy(sim, axis=1) == 100.0
    assert retrieval_accuracy(sim, axis=0) == 100.0


def test_asymmetric_retrieval():
    """
    Text-to-image and image-to-text accuracies are computed separately.
    """
    sim = np.array([[0.9, 0.1],
                    [0.95, 0.2]])
    assert retrieval_accuracy(sim, axis=1) == 50.0
    assert retrieval_accuracy(sim, axis=0) == 50.0


def test_diagonal_margin():
    """
    The margin is the diagonal minus the off-diagonal row mean.
    """
    sim = np.array([[0.5, 0.1, 0.3],
                    [0.2, 0.4, 0.2],
                    [0.0, 0.0, 0.6]])
    expected = np.mean([0.5 - 0.2, 0.4 - 0.2, 0.6 - 0.0])
    assert np.isclose(diagonal_margin(sim), expected)


def test_compare_models_sorted():
    """
    The comparison table is sorted by average accuracy.
    """
    good = np.eye(3)
    bad = np.ones((3, 3)) - np.eye(3)
    table = compare_models({"bad": bad, "good": good})
    assert list(table["Model"]) == ["good", "bad"]
    assert set(evaluate_similarity(good)) <= set(table.columns)
