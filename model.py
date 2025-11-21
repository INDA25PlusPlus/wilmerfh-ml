from nn import Linear
from tensor import Tensor


class MLP:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden: int = 2,
        hidden_size: int = 128,
    ):
        self.input_dim, self.output_dim, self.num_hidden, self.hidden_size = (
            input_dim,
            output_dim,
            num_hidden,
            hidden_size,
        )
        self.input_layer = Linear(input_dim, hidden_size)
        assert num_hidden > 0, "Number of hidden layers must be positive"
        self.hidden_layers = [
            Linear(hidden_size, hidden_size) for _ in range(num_hidden - 1)
        ]
        self.output_layer = Linear(hidden_size, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        ret = self.input_layer.forward(x)
        ret = ret.lrelu()
        for layer in self.hidden_layers:
            ret = layer.forward(ret)
            ret = ret.lrelu()
        output = self.output_layer.forward(ret)
        return output

    def parameters(self) -> list[Tensor]:
        params = []
        params.extend(self.input_layer.parameters())
        for layer in self.hidden_layers:
            params.extend(layer.parameters())
        params.extend(self.output_layer.parameters())
        return params

    def state_dict(self) -> dict:
        state = {
            "input_layer": self.input_layer.state_dict(),
            "hidden_layers": [layer.state_dict() for layer in self.hidden_layers],
            "output_layer": self.output_layer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: dict):
        self.input_layer.load_state_dict(state_dict["input_layer"])
        for i, layer_state in enumerate(state_dict["hidden_layers"]):
            self.hidden_layers[i].load_state_dict(layer_state)
        self.output_layer.load_state_dict(state_dict["output_layer"])
