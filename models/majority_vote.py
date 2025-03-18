"""Majority Vote model to combine the outputs"""

import torch
from torch import Tensor, nn


class MajorityVote(nn.Module):
    """Majority Vote model to combine the outputs.
    NOTE: because this model does not lazy load models for evaluation, you should evaluate it on CPU.
    """

    def __init__(self, models: list[nn.Module], n_classes=8):
        """Build a Majority Vote model from a list of models, without weight discrimination."""
        super().__init__()

        self.models = models
        self.weights = [1.0] * len(models)
        self.n_classes = n_classes

    @classmethod
    def from_weights(cls, models: list[nn.Module], weights: list[float]):
        """Build a Majority Vote model from a list of models and weights (usually their respective accuracies)."""
        instance = cls(models)
        instance.weights = weights
        return instance

    def forward(self, x: Tensor):

        results = torch.zeros((len(x), self.n_classes), device=x.device)

        for coeff, model in zip(self.weights, self.models):
            results += model(x) * coeff

        return results
