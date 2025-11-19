from tensor import Tensor


class SGD:
    def __init__(self, parameters: list[Tensor], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None


def test_sgd_basic_update():
    param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.1, 0.1]).data
    optimizer = SGD([param], learning_rate=0.1)
    optimizer.step()
    expected = Tensor([0.99, 1.99, 2.99]).data
    assert (param.data == expected).all()


def test_sgd_zero_grad():
    param1 = Tensor([1.0], requires_grad=True)
    param2 = Tensor([2.0], requires_grad=True)
    param1.grad = Tensor([0.5]).data
    param2.grad = Tensor([1.0]).data
    optimizer = SGD([param1, param2], learning_rate=0.01)
    optimizer.zero_grad()
    assert param1.grad is None
    assert param2.grad is None
