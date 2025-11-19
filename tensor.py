import numpy as np


class Add:
    def __init__(self):
        self.a, self.b = None, None

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        self.a, self.b = a, b
        data = a.data + b.data
        ret = Tensor(data)
        ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return grad, grad


class Mul:
    def __init__(self):
        self.a, self.b = None, None

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        self.a, self.b = a, b
        data = a.data * b.data
        ret = Tensor(data)
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
        ret = Tensor(data)
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
        ret = Tensor(data)
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
        self.a, self.b = a, b
        data = a.data @ b.data
        ret = Tensor(data)
        ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.a is not None and self.b is not None
        grad_a = grad @ self.b.data.T
        grad_b = self.a.data.T @ grad
        return grad_a, grad_b


BINARY_OP = Add | Mul | MatMul
UNARY_OP = Neg | LeakyReLU


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.grad: np.ndarray | None = None
        self.src: BINARY_OP | UNARY_OP | None = None

    @classmethod
    def randn(cls, *shape, scale=1.0) -> "Tensor":
        return cls(np.random.randn(*shape) * scale)

    @classmethod
    def zeros(cls, *shape) -> "Tensor":
        return cls(np.zeros(*shape))

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

    def lrelu(self, negative_slope=0.01):
        return LeakyReLU(negative_slope).forward(self)

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        if self.src is not None:
            if isinstance(self.src, BINARY_OP):
                assert self.src.a is not None and self.src.b is not None
                grad_a, grad_b = self.src.backward(self.grad)
                self.src.a.grad = (
                    grad_a if self.src.a.grad is None else self.src.a.grad + grad_a
                )
                self.src.b.grad = (
                    grad_b if self.src.b.grad is None else self.src.b.grad + grad_b
                )
                self.src.a.backward()
                self.src.b.backward()
            elif isinstance(self.src, UNARY_OP):
                assert self.src.a is not None
                grad_a = self.src.backward(self.grad)
                self.src.a.grad = (
                    grad_a if self.src.a.grad is None else self.src.a.grad + grad_a
                )
                self.src.a.backward()


def test_tensor_addition():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    expected = np.array([5.0, 7.0, 9.0])
    assert np.allclose(c.data, expected)


def test_tensor_multiplication():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [1.0, 2.0]])
    c = a * b
    expected = np.array([[2.0, 0.0], [3.0, 8.0]])
    assert np.allclose(c.data, expected)


def test_leaky_relu_forward():
    x = Tensor([[-1.0, 0.5], [-2.0, 1.0]])
    y = x.lrelu()
    expected = np.array([[-0.01, 0.5], [-0.02, 1.0]])
    assert np.allclose(y.data, expected)


def test_leaky_relu_gradients():
    x = Tensor([[-1.0, 0.5], [-2.0, 1.0]])
    y = x.lrelu()
    y.backward()
    expected_grad = np.array([[0.01, 1.0], [0.01, 1.0]])
    assert np.allclose(x.grad, expected_grad)


def test_matrix_multiplication():
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(C.data, expected)


def test_complex_computation():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[0.5, 1.0], [1.5, 2.0]])
    c = Tensor([[2.0, 0.0], [0.0, 2.0]])
    sum_ab = a + b
    result = sum_ab * c
    expected = np.array([[3.0, 0.0], [0.0, 12.0]])
    assert np.allclose(result.data, expected)
    result.backward()
    expected_grad_a = np.array([[2.0, 0.0], [0.0, 2.0]])
    expected_grad_b = np.array([[2.0, 0.0], [0.0, 2.0]])
    expected_grad_c = np.array([[1.5, 3.0], [4.5, 6.0]])
    assert np.allclose(a.grad, expected_grad_a)
    assert np.allclose(b.grad, expected_grad_b)
    assert np.allclose(c.grad, expected_grad_c)
