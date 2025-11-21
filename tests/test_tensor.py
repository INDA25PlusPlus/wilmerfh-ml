import numpy as np
import torch

from tensor import Tensor


def test_addition_forward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    result = a + b
    a_torch = torch.tensor([1.0, 2.0, 3.0])
    b_torch = torch.tensor([4.0, 5.0, 6.0])
    expected = torch.add(a_torch, b_torch)
    assert np.allclose(result.data, expected.numpy())


def test_addition_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    result = a + b
    result.sum().backward()
    a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    expected = a_torch + b_torch
    expected.sum().backward()
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_multiplication_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [1.0, 2.0]])
    result = a * b
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_torch = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    expected = torch.mul(a_torch, b_torch)
    assert np.allclose(result.data, expected.numpy())


def test_multiplication_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
    result = a * b
    result.sum().backward()
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_torch = torch.tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
    expected = a_torch * b_torch
    expected.sum().backward()
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_leaky_relu_forward():
    a = Tensor([[-1.0, 0.5], [-2.0, 1.0]])
    result = a.lrelu(negative_slope=0.01)
    a_torch = torch.tensor([[-1.0, 0.5], [-2.0, 1.0]])
    expected = torch.nn.functional.leaky_relu(a_torch, negative_slope=0.01)
    assert np.allclose(result.data, expected.numpy())


def test_leaky_relu_backward():
    a = Tensor([[-1.0, 0.5], [-2.0, 1.0]], requires_grad=True)
    result = a.lrelu()
    result.sum().backward()

    a_torch = torch.tensor([[-1.0, 0.5], [-2.0, 1.0]], requires_grad=True)
    expected = torch.nn.functional.leaky_relu(a_torch, negative_slope=0.01)
    expected.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_matrix_multiplication_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    result = a @ b
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected = torch.matmul(a_torch, b_torch)
    assert np.allclose(result.data, expected.numpy())


def test_matrix_multiplication_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    result = a @ b
    result.sum().backward()
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    expected = a_torch @ b_torch
    expected.sum().backward()
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_complex_computation():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
    c = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    sum_ab = a + b
    result = sum_ab * c

    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_torch = torch.tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
    c_torch = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    sum_ab_torch = a_torch + b_torch
    result_torch = sum_ab_torch * c_torch

    assert np.allclose(result.data, result_torch.detach().numpy())

    result.sum().backward()
    result_torch.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())
    assert np.allclose(c.grad, c_torch.grad.numpy())


def test_chained_ops():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
    c = Tensor([[0.0, 1.0], [1.0, 0.5]], requires_grad=True)

    matmul_result = a @ b
    activated = matmul_result.lrelu()
    exp_c = c.exp()
    summed = activated + exp_c
    result = summed.sum()

    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_torch = torch.tensor([[0.5, 1.0], [1.5, 2.0]], requires_grad=True)
    c_torch = torch.tensor([[0.0, 1.0], [1.0, 0.5]], requires_grad=True)

    matmul_torch = a_torch @ b_torch
    activated_torch = torch.nn.functional.leaky_relu(matmul_torch, negative_slope=0.01)
    exp_c_torch = torch.exp(c_torch)
    summed_torch = activated_torch + exp_c_torch
    expected = summed_torch.sum()

    assert np.allclose(result.data, expected.detach().numpy())

    result.backward()
    expected.backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())
    assert np.allclose(c.grad, c_torch.grad.numpy())


def test_requires_grad_inheritance():
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=False)
    c = a + b
    assert c.requires_grad == True
    assert c.src is not None
    c.backward()
    assert a.grad is not None
    assert b.grad is None


def test_exp_forward():
    a = Tensor([0.0, 1.0, 2.0])
    result = a.exp()
    a_torch = torch.tensor([0.0, 1.0, 2.0])
    expected = torch.exp(a_torch)
    assert np.allclose(result.data, expected.numpy())


def test_exp_backward():
    a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
    result = a.exp()
    result.sum().backward()

    a_torch = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    expected = torch.exp(a_torch)
    expected.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_max_forward():
    a = Tensor([[1.0, 5.0], [3.0, 2.0]])
    result1 = a.max(axis=0)
    a_torch = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    expected1 = torch.max(a_torch, dim=0).values
    assert np.allclose(result1.data, expected1.numpy())

    result2 = a.max(axis=1)
    expected2 = torch.max(a_torch, dim=1).values
    assert np.allclose(result2.data, expected2.numpy())


def test_max_backward():
    a1 = Tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    result1 = a1.max(axis=0)
    result1.sum().backward()

    a1_torch = torch.tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
    expected1 = torch.max(a1_torch, dim=0).values
    expected1.sum().backward()

    assert np.allclose(a1.grad, a1_torch.grad.numpy())

    a2 = Tensor([[1.0, 5.0], [3.0, 5.0]], requires_grad=True)
    result2 = a2.max(axis=1)
    result2.sum().backward()

    a2_torch = torch.tensor([[1.0, 5.0], [3.0, 5.0]], requires_grad=True)
    expected2 = torch.max(a2_torch, dim=1).values
    expected2.sum().backward()

    assert np.allclose(a2.grad, a2_torch.grad.numpy())


def test_sum_forward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y1 = x.sum(axis=0)
    expected1 = torch.sum(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=0)
    assert np.allclose(y1.data, expected1.numpy())

    y2 = x.sum(axis=1)
    expected2 = torch.sum(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=1)
    assert np.allclose(y2.data, expected2.numpy())


def test_sum_backward():
    x1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y1 = x1.sum(axis=0)
    y1.sum().backward()

    x1_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y1_torch = torch.sum(x1_torch, dim=0)
    y1_torch.sum().backward(retain_graph=True)
    grad1_torch = torch.autograd.grad(y1_torch.sum(), x1_torch)[0]

    assert np.allclose(x1.grad, grad1_torch.numpy())

    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2 = x2.sum(axis=1)
    y2.sum().backward()

    x2_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2_torch = torch.sum(x2_torch, dim=1)
    grad2_torch = torch.autograd.grad(y2_torch.sum(), x2_torch)[0]

    assert np.allclose(x2.grad, grad2_torch.numpy())


def test_div_forward():
    a = Tensor([4.0, 6.0, 8.0])
    b = Tensor([2.0, 3.0, 4.0])
    result = a / b
    a_torch = torch.tensor([4.0, 6.0, 8.0])
    b_torch = torch.tensor([2.0, 3.0, 4.0])
    expected = torch.div(a_torch, b_torch)
    assert np.allclose(result.data, expected.numpy())


def test_div_backward():
    a = Tensor([4.0, 6.0], requires_grad=True)
    b = Tensor([2.0, 3.0], requires_grad=True)
    result = a / b
    result.sum().backward()

    a_torch = torch.tensor([4.0, 6.0], requires_grad=True)
    b_torch = torch.tensor([2.0, 3.0], requires_grad=True)
    expected = torch.div(a_torch, b_torch)
    expected.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_broadcast_forward():
    a = Tensor([1.0, 2.0])
    result1 = a.broadcast_to((3, 2))
    assert result1.data.shape == (3, 2)
    assert np.allclose(result1.data, np.broadcast_to(a.data, (3, 2)))

    b = Tensor([0.5])
    result2 = b.broadcast_to((2, 3))
    assert result2.data.shape == (2, 3)
    assert np.allclose(result2.data, np.full((2, 3), 0.5))


def test_broadcast_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    broadcasted = a.broadcast_to((3, 2))
    broadcasted.sum().backward()
    expected_grad = np.array([3.0, 3.0])
    assert np.allclose(a.grad, expected_grad)

    b = Tensor([0.5], requires_grad=True)
    broadcasted2 = b.broadcast_to((2, 3))
    broadcasted2.sum().backward()
    expected_grad2 = np.array([6.0])
    assert np.allclose(b.grad, expected_grad2)


def test_power_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = a**2.0
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = a_torch**2.0
    assert np.allclose(result.data, expected.numpy())


def test_power_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    result = a**2.0
    result.sum().backward()
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    expected = a_torch**2.0
    expected.sum().backward()
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_log_forward():
    a = Tensor([1.0, np.e, np.e**2])
    result = a.log()
    a_torch = torch.tensor([1.0, np.e, np.e**2])
    expected = torch.log(a_torch)
    assert np.allclose(result.data, expected.numpy())


def test_log_backward():
    a = Tensor([1.0, np.e, np.e**2], requires_grad=True)
    result = a.log()
    result.sum().backward()

    a_torch = torch.tensor([1.0, np.e, np.e**2], requires_grad=True)
    expected = torch.log(a_torch)
    expected.sum().backward()

    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_mean_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = a.mean()
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.mean(a_torch)
    assert np.allclose(result.data, expected.numpy())

    result_axis = a.mean(axis=0)
    expected_axis = torch.mean(a_torch, dim=0)
    assert np.allclose(result_axis.data, expected_axis.numpy())


def test_mean_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    result = a.mean()
    result.backward()

    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    expected = a_torch.mean()
    expected.backward()
    assert np.allclose(a.grad, a_torch.grad.numpy())

    a_axis = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    result_axis = a_axis.mean(axis=0)
    result_axis.sum().backward()

    a_axis_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    expected_axis = a_axis_torch.mean(dim=0)
    expected_axis.sum().backward()
    assert np.allclose(a_axis.grad, a_axis_torch.grad.numpy())
