import torch
from torch.nn import functional as F

class LossOnStep(torch.nn.Module):
    """
    Allows computation of an existing loss function on only one step of a sequence of outputs.
    """
    def __init__(self, loss: torch.nn.Module, time_dim: int, step: int):
        super().__init__()
        self.loss = loss
        self.time_slice = slice(step * time_dim, (step + 1) * time_dim)

    def forward(self, inputs, targets):
        return self.loss(inputs[:, self.time_slice], targets[:, self.time_slice])
