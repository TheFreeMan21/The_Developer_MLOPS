import torch

from src.models.model import SimpleCNN


def test_simplecnn_outputs_one_logit_per_image():
    model = SimpleCNN()
    batch = torch.randn(4, 3, 64, 64)

    output = model(batch)

    assert output.shape == (4, 1)