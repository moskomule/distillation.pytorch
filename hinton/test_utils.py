import torch

from utils import hot_logsoftmax


def test_hot_logsoftmax():
    # if temp == 1, hot_logsoftmax is log_softmax
    input = torch.randn(4, 3)
    assert all((hot_logsoftmax(input, temperature=1, dim=-1) == input.log_softmax(dim=-1)).view(-1))
