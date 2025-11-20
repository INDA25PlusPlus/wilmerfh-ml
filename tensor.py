import numpy as np
import torch


class Add:
    def __init__(self):
        self.a, self.b = None, None

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        if a.data.shape != b.data.shape:
            raise ValueError(f"Shape mismatch: {a.data.shape} vs {b.data.shape}")
        self.a, self.b = a, b
        data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        ret = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return grad, grad


class Mul:
    def __init__(self):
        self.a, self.b = None, None

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        if a.data.shape != b.data.shape:
            raise ValueError(f"Shape mismatch: {a.data.shape} vs {b.data.shape}")
        self.a, self.b = a, b
        data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        ret = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.a is not None and self.b is not None
        return grad * self.b.data, grad * self.a.data


class Neg:
    def __init__(self):
        self.a = None

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        data = -a.data
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        return -grad


class LeakyReLU:
    def __init__(self, negative_slope=0.01):
        self.a = None
        self.negative_slope = negative_slope

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        data = np.where(a.data > 0, a.data, self.negative_slope * a.data)
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        mask = np.where(self.a.data > 0, 1.0, self.negative_slope)
        return grad * mask


class MatMul:
    def __init__(self):
        self.a, self.b = None, None

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        if a.data.shape[1] != b.data.shape[0]:
            raise ValueError(f"MatMul shape mismatch: {a.data.shape} @ {b.data.shape}")
        self.a, self.b = a, b
        data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        ret = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.a is not None and self.b is not None
        grad_a = grad @ self.b.data.T
        grad_b = self.a.data.T @ grad
        return grad_a, grad_b


class Exp:
    def __init__(self):
        self.a = None

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        data = np.exp(a.data)
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        return grad * np.exp(self.a.data)


class Max:
    def __init__(self, axis=None):
        self.a = None
        self.axis = axis

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        data = np.max(a.data, axis=self.axis)
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        a_data = self.a.data
        if self.axis is None:
            max_val = np.max(a_data)
            mask = (a_data == max_val).astype(float)
            mask_sum = np.sum(mask)
            mask = mask / np.where(mask_sum == 0, 1, mask_sum)
            return grad * mask

        max_data = np.max(a_data, axis=self.axis, keepdims=True)
        mask = (a_data == max_data).astype(float)
        mask_sum = np.sum(mask, axis=self.axis, keepdims=True)
        mask = mask / np.where(mask_sum == 0, 1, mask_sum)
        grad_expanded = np.expand_dims(grad, axis=self.axis)
        return grad_expanded * mask


class Sum:
    def __init__(self, axis=None):
        self.a = None
        self.axis = axis

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        self.original_shape = a.data.shape
        data = np.sum(a.data, axis=self.axis)
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        if self.axis is None:
            return np.full(self.a.data.shape, grad)

        grad_expanded = np.expand_dims(grad, axis=self.axis)
        return np.broadcast_to(grad_expanded, self.a.data.shape)


class Div:
    def __init__(self):
        self.a, self.b = None, None

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        if a.data.shape != b.data.shape:
            raise ValueError(f"Shape mismatch: {a.data.shape} vs {b.data.shape}")
        self.a, self.b = a, b
        data = a.data / b.data
        requires_grad = a.requires_grad or b.requires_grad
        ret = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.a is not None and self.b is not None
        grad_a = grad / self.b.data
        grad_b = -grad * self.a.data / (self.b.data**2)
        return grad_a, grad_b


BINARY_OP = Add | Mul | MatMul | Div
UNARY_OP = Neg | LeakyReLU | Exp | Max | Sum


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.grad: np.ndarray | None = None
        self.src: BINARY_OP | UNARY_OP | None = None
        self.requires_grad = requires_grad

    @classmethod
    def randn(cls, *shape, scale=1.0, requires_grad=False) -> "Tensor":
        return cls(np.random.randn(*shape) * scale, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, *shape, requires_grad=False) -> "Tensor":
        return cls(np.zeros(*shape), requires_grad=requires_grad)

    def __add__(self, other: "Tensor"):
        return Add().forward(self, other)

    def __neg__(self):
        return Neg().forward(self)

    def __sub__(self, other: "Tensor"):
        return self + (-other)

    def __mul__(self, other: "Tensor"):
        return Mul().forward(self, other)

    def __matmul__(self, other: "Tensor"):
        return MatMul().forward(self, other)

    def __truediv__(self, other: "Tensor"):
        return Div().forward(self, other)

    def lrelu(self, negative_slope=0.01):
        return LeakyReLU(negative_slope).forward(self)

    def exp(self):
        return Exp().forward(self)

    def max(self, axis=None):
        return Max(axis=axis).forward(self)

    def sum(self, axis=None):
        return Sum(axis=axis).forward(self)

    def div(self, other: "Tensor"):
        return Div().forward(self, other)

    def backward(self):
        if not self.requires_grad:
            return

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        if self.src is not None:
            if isinstance(self.src, BINARY_OP):
                assert self.src.a is not None and self.src.b is not None
                grad_a, grad_b = self.src.backward(self.grad)
                if self.src.a.requires_grad:
                    self.src.a.grad = (
                        grad_a if self.src.a.grad is None else self.src.a.grad + grad_a
                    )
                    self.src.a.backward()
                if self.src.b.requires_grad:
                    self.src.b.grad = (
                        grad_b if self.src.b.grad is None else self.src.b.grad + grad_b
                    )
                    self.src.b.backward()
            elif isinstance(self.src, UNARY_OP):
                assert self.src.a is not None
                grad_a = self.src.backward(self.grad)
                if self.src.a.requires_grad:
                    self.src.a.grad = (
                        grad_a if self.src.a.grad is None else self.src.a.grad + grad_a
                    )
                    self.src.a.backward()


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

    assert np.allclose(a.grad, a_torch.grad.numpy(), atol=1e-6)


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

    assert np.allclose(a.grad, a_torch.grad.numpy(), atol=1e-6)
    assert np.allclose(b.grad, b_torch.grad.numpy(), atol=1e-6)
    assert np.allclose(c.grad, c_torch.grad.numpy(), atol=1e-6)


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
