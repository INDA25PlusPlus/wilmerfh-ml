from optim import SGD
from tensor import Tensor


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
