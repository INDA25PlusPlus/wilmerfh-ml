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
