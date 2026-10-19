"""Explicit aleatoric uncertainty head; simple diagonal Gaussian baseline."""
import math
import torch
from torch import nn


class DiagonalGaussian(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mean = nn.Linear(in_dim, out_dim)
        self.log_variance = nn.Linear(in_dim, out_dim)

    def forward(self, features):
        mu = self.mean(features)
        logvar = torch.clamp(self.log_variance(features), -12, 8)
        return mu, logvar

    @staticmethod
    def negative_log_likelihood(target, mean, logvar):
        return (0.5 * (math.log(2.0 * math.pi) + logvar +
                       (target - mean).square() / torch.exp(logvar))).mean()

    @staticmethod
    def sample(mean, logvar):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
