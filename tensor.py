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
UNARY_OP = Neg


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        self.grad: np.ndarray | None = None
        self.src: BINARY_OP | UNARY_OP | None = None

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
