import copy

import numpy as np

from tensor import Tensor


class Linear:
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        self.input_dim, self.output_dim, self.bias = input_dim, output_dim, bias
        self.weights = Tensor.randn(
            input_dim, output_dim, scale=0.01, requires_grad=True
        )
        self.bias_tensor = (
            Tensor.randn(output_dim, scale=0.01, requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weights
        if self.bias_tensor is not None:
            batch_size = x.data.shape[0]
            target_shape = (batch_size, self.output_dim)
            bias_broadcast = self.bias_tensor.broadcast_to(target_shape)
            output = output + bias_broadcast
        return output

    def parameters(self) -> list[Tensor]:
        params = [self.weights]
        if self.bias_tensor is not None:
            params.append(self.bias_tensor)
        return params

    def state_dict(self) -> dict[str, Tensor]:
        state = {"weights": copy.deepcopy(self.weights)}
        if self.bias_tensor is not None:
            state["bias"] = copy.deepcopy(self.bias_tensor)
        return state

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.weights = copy.deepcopy(state_dict["weights"])
        if "bias" in state_dict:
            self.bias_tensor = copy.deepcopy(state_dict["bias"])
        else:
            self.bias_tensor = None
