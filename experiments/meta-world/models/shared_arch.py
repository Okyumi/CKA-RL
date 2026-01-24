import math
import torch
import torch.nn as nn


def lecun_uniform_init(tensor):
    """LeCun uniform initialization: variance_scaling(1/3, "fan_in", "uniform")"""
    fan_in = tensor.size(1)
    bound = math.sqrt(1.0 / (3.0 * fan_in))
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


class ResidualBlock(nn.Module):
    """Residual block with 4 linear layers."""
    def __init__(self, width: int, use_layer_norm: bool = True, use_relu: bool = False):
        super().__init__()
        self.activation = nn.ReLU() if use_relu else nn.SiLU()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(4)])
        if use_layer_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        else:
            self.norms = nn.ModuleList([nn.Identity() for _ in range(4)])
        for layer in self.layers:
            lecun_uniform_init(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
        return x + identity


class SharedTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int = 1024,
        depth: int = 4,
        use_layer_norm: bool = True,
        use_relu: bool = False,
    ):
        super().__init__()
        self.output_dim = width
        self.activation = nn.ReLU() if use_relu else nn.SiLU()
        self.input_layer = nn.Linear(input_dim, width)
        self.input_norm = nn.LayerNorm(width) if use_layer_norm else nn.Identity()
        lecun_uniform_init(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        num_blocks = max(1, depth // 4)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, use_layer_norm, use_relu) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        return x


def shared(
    input_dim: int,
    network_width: int = 1024,
    network_depth: int = 4,
    use_layer_norm: bool = True,
    use_relu: bool = False,
):
    return SharedTrunk(
        input_dim=input_dim,
        width=network_width,
        depth=network_depth,
        use_layer_norm=use_layer_norm,
        use_relu=use_relu,
    )
