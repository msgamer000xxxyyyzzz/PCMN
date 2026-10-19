"""Mirror-style learning = shared action/observation embedding, not biological neurons."""
import torch
from torch import nn
from torch.nn import functional as F


class MirrorLearning(nn.Module):
    def __init__(self, observed_dim: int, action_dim: int, latent_dim: int = 64):
        super().__init__()
        self.observe = nn.Sequential(nn.Linear(observed_dim, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, latent_dim))
        self.act = nn.Sequential(nn.Linear(action_dim, latent_dim), nn.SiLU(),
                                 nn.Linear(latent_dim, latent_dim))
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))

    def forward(self, observed, actions):
        z_obs = F.normalize(self.observe(observed), dim=-1)
        z_act = F.normalize(self.act(actions), dim=-1)
        return z_obs, z_act

    def paired_loss(self, observed, actions):
        """Paired samples must correspond, use batch >=2 for negative examples."""
        z_obs, z_act = self.forward(observed, actions)
        if z_obs.size(0) < 2:
            raise ValueError('Paired contrastive loss requires batch >= 2')
        temperature = torch.exp(self.log_temperature).clamp(0.02, 2)
        scores = z_obs @ z_act.T / temperature
        targets = torch.arange(scores.size(0), device=scores.device)
        return 0.5 * (F.cross_entropy(scores, targets) +
                      F.cross_entropy(scores.T, targets))
