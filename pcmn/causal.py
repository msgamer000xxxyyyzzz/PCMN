"""An explicit, ordered linear structural causal model (SCM).

Correct use requires the causal ordering and an appropriate graph to be known
or separately estimated from observational AND suitable interventional data.
A trained SCM is not automatically a discovery of real-world causes.
"""
import torch
from torch import nn


class OrderedLinearSCM(nn.Module):
    """x[i] = sum_{j < i} W[i,j] * x[j] + epsilon[i], with optional do() fixes."""
    def __init__(self, n_variables: int):
        super().__init__()
        if n_variables < 2:
            raise ValueError('At least two variables are required')
        self.n_variables = n_variables
        self.weights = nn.Parameter(torch.zeros(n_variables, n_variables))
        self.register_buffer('order_mask', torch.tril(torch.ones(n_variables, n_variables), diagonal=-1))

    def forward(self, independent_noise, interventions=None):
        """independent_noise [batch,n_variables].

        interventions: {variable_index: number_or_[batch] tensor}, simulating do(X=value)
        under the *assumed* structural equations.
        """
        if independent_noise.ndim != 2 or independent_noise.size(1) != self.n_variables:
            raise ValueError('noise must have shape [batch,n_variables]')
        if interventions is None:
            interventions = {}
        for key in interventions:
            if not isinstance(key, int) or not (0 <= key < self.n_variables):
                raise ValueError('Invalid intervention variable index')
        W = self.weights * self.order_mask
        values = []
        for i in range(self.n_variables):
            if i in interventions:
                val = torch.as_tensor(interventions[i], dtype=independent_noise.dtype,
                                      device=independent_noise.device)
                value = val + torch.zeros_like(independent_noise[:, i])
            else:
                value = independent_noise[:, i]
                if i:
                    parents = torch.stack(values, dim=1)
                    value = value + (parents * W[i, :i]).sum(dim=1)
            values.append(value)
        return torch.stack(values, dim=1)
