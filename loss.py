import numpy as np
import torch

from tensor import Tensor


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    assert prediction.data.shape == target.data.shape, (
        f"Shape mismatch: {prediction.data.shape} vs {target.data.shape}"
    )

    diff = prediction - target
    squared = diff**2.0
    sum_squared = squared.sum()
    n = Tensor(prediction.data.size)
    loss = sum_squared / n

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

    batch_size = prediction.data.shape[0]
    log_softmax_output = prediction.log_softmax(axis=-1)
    selected_log_probs = log_softmax_output * target
    summed_log_probs = selected_log_probs.sum()
    loss = -summed_log_probs / Tensor(batch_size)

    return loss


def test_mse_loss_forward():
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[1.5, 1.8], [2.5, 4.5]])
    loss = mse_loss(pred, target)

    pred_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    target_torch = torch.tensor([[1.5, 1.8], [2.5, 4.5]], dtype=torch.float32)
    expected = torch.nn.functional.mse_loss(pred_torch, target_torch)

    assert np.allclose(loss.data, expected.numpy())


def test_mse_loss_backward():
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = Tensor([[1.5, 1.8], [2.5, 4.5]])
    loss = mse_loss(pred, target)
    loss.backward()

    pred_torch = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
    )
    target_torch = torch.tensor([[1.5, 1.8], [2.5, 4.5]], dtype=torch.float32)
    expected = torch.nn.functional.mse_loss(pred_torch, target_torch)
    expected.backward()

    assert np.allclose(pred.grad, pred_torch.grad.numpy())


def test_cross_entropy_loss_forward():
    pred = Tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], requires_grad=True)
    target_labels = np.array([2, 1])
    num_classes = pred.data.shape[1]
    target = Tensor(np.eye(num_classes)[target_labels.astype(int)])

    loss = cross_entropy_loss(pred, target)

    pred_torch = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], dtype=torch.float32)
    target_torch = torch.tensor([2, 1], dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(pred_torch, target_torch)

    assert np.allclose(loss.data, expected.numpy())


def test_cross_entropy_loss_backward():
    pred = Tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], requires_grad=True)
    target_labels = np.array([2, 1])
    num_classes = pred.data.shape[1]
    target = Tensor(np.eye(num_classes)[target_labels.astype(int)])

    loss = cross_entropy_loss(pred, target)
    loss.backward()

    pred_torch = torch.tensor(
        [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], dtype=torch.float32, requires_grad=True
    )
    target_torch = torch.tensor([2, 1], dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(pred_torch, target_torch)
    expected.backward()

    assert pred.grad is not None
    assert pred_torch.grad is not None
    assert np.allclose(pred.grad, pred_torch.grad.numpy())
