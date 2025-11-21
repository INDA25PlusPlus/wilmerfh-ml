import numpy as np

from nn import Linear
from tensor import Tensor


def test_linear_layer_forward():
    layer = Linear(input_dim=3, output_dim=2)
    x = Tensor([[1.0, 2.0, 3.0]])
    output = layer.forward(x)
    assert output.data.shape == (1, 2)


def test_linear_layer_gradients():
    layer = Linear(input_dim=2, output_dim=1)
    layer.weights.data = np.array([[1.0], [2.0]])
    assert layer.bias_tensor is not None
    layer.bias_tensor.data = np.array([0.5])
    x = Tensor([[3.0, 4.0]], requires_grad=True)
    output = layer.forward(x)
    expected_output = np.array([[3.0 * 1.0 + 4.0 * 2.0 + 0.5]])
    assert np.allclose(output.data, expected_output)
    output.backward()
    expected_weight_grad = np.array([[3.0], [4.0]])
    expected_bias_grad = np.array([1.0])
    assert layer.weights.grad is not None
    assert layer.bias_tensor.grad is not None
    assert np.allclose(layer.weights.grad, expected_weight_grad)
    assert np.allclose(layer.bias_tensor.grad, expected_bias_grad)
