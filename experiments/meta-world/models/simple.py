import torch
import torch.nn as nn
import os
from .shared_arch import shared


class SimpleAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, network_width: int = 1024, network_depth: int = 4):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.fc = shared(
            input_dim=obs_dim,
            network_width=network_width,
            network_depth=network_depth,
        )
        self.fc_output_dim = getattr(self.fc, "output_dim", 256)

        # will be created when calling `reset_heads`
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = nn.Linear(self.fc_output_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.fc_output_dim, self.act_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model
