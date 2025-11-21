import numpy as np

from tensor import Tensor


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    assert prediction.data.shape == target.data.shape, (
        f"Shape mismatch: {prediction.data.shape} vs {target.data.shape}"
    )

    diff = prediction - target
    squared = diff**2.0
    loss = squared.mean()

    return loss


def cross_entropy_loss(prediction: Tensor, target: Tensor) -> Tensor:
    assert prediction.data.ndim == 2, (
        f"Prediction should be 2D, but got {prediction.data.ndim}D"
    )
    assert target.data.ndim == 2, (
        f"Target should be 2D (one-hot), but got {target.data.ndim}D"
    )
    assert prediction.data.shape == target.data.shape, (
        "Prediction and target shapes must match: "
        f"{prediction.data.shape} vs {target.data.shape}"
    )

    log_softmax_output = prediction.log_softmax(axis=-1)
    selected_log_probs = log_softmax_output * target
    loss = -selected_log_probs.sum(axis=1).mean()

    return loss
