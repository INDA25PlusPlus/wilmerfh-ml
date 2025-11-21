import numpy as np


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


class LogSoftmax:
    def __init__(self, axis=-1):
        self.a = None
        self.axis = axis
        self.softmax_output = None

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        max_val = np.max(a.data, axis=self.axis, keepdims=True)
        exp_data = np.exp(a.data - max_val)
        sum_exp_data = np.sum(exp_data, axis=self.axis, keepdims=True)
        self.softmax_output = exp_data / sum_exp_data
        log_softmax_data = a.data - max_val - np.log(sum_exp_data)

        ret = Tensor(log_softmax_data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None and self.softmax_output is not None
        sum_grad = np.sum(grad, axis=self.axis, keepdims=True)
        return grad - sum_grad * self.softmax_output


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


class Broadcast:
    def __init__(self, target_shape):
        self.a = None
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        self.original_shape = a.data.shape
        data = np.broadcast_to(a.data, self.target_shape)
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        result = grad
        ndim_diff = grad.ndim - len(self.original_shape)
        for _ in range(ndim_diff):
            result = np.sum(result, axis=0)

        for i, (orig_dim, grad_dim) in enumerate(
            zip(self.original_shape, result.shape)
        ):
            if orig_dim == 1 and grad_dim > 1:
                result = np.sum(result, axis=i, keepdims=True)

        return result


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


class Power:
    def __init__(self):
        self.a = None
        self.exponent = None

    def forward(self, a: "Tensor", exponent: float) -> "Tensor":
        self.a = a
        self.exponent = exponent
        data = a.data**exponent
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        return grad * self.exponent * (self.a.data ** (self.exponent - 1))


class Log:
    def __init__(self):
        self.a = None

    def forward(self, a: "Tensor") -> "Tensor":
        self.a = a
        data = np.log(a.data)
        ret = Tensor(data, requires_grad=a.requires_grad)
        if a.requires_grad:
            ret.src = self
        return ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.a is not None
        return grad / self.a.data


BINARY_OP = Add | Mul | MatMul | Div
UNARY_OP = Neg | LeakyReLU | Exp | Max | Sum | Broadcast | Power | LogSoftmax | Log


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

    def __pow__(self, exponent: float):
        return Power().forward(self, exponent)

    def lrelu(self, negative_slope=0.01):
        return LeakyReLU(negative_slope).forward(self)

    def log_softmax(self, axis=-1):
        return LogSoftmax(axis).forward(self)

    def log(self):
        return Log().forward(self)

    def exp(self):
        return Exp().forward(self)

    def max(self, axis=None):
        return Max(axis=axis).forward(self)

    def sum(self, axis=None):
        return Sum(axis=axis).forward(self)

    def mean(self, axis=None):
        val = self.sum(axis=axis)
        num_elements = self.data.size if axis is None else self.data.shape[axis]
        return val / Tensor(np.full(val.data.shape, num_elements))

    def div(self, other: "Tensor"):
        return Div().forward(self, other)

    def broadcast_to(self, target_shape):
        return Broadcast(target_shape).forward(self)

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
