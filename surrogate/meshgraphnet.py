"""MeshGraphNet Encode-Process-Decode model (augmented-graph ready).

Paper-faithful MeshGraphNet: encoder MLPs with a
single trailing LayerNorm, K message-passing steps with independent parameters
and residual connections, decoder MLP without LayerNorm. Works on any node/edge
feature dimensions, so it operates directly on the *augmented* graph.
"""
from __future__ import annotations

import torch
from torch import nn


def build_mlp(input_size: int, output_size: int, hidden_size: int = 128,
              num_hidden_layers: int = 2, layer_norm: bool = False) -> nn.Sequential:
    """MLP with `num_hidden_layers` hidden layers, ReLU and a single optional
    LayerNorm after the final layer (paper / DeepMind / NVIDIA convention)."""
    layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
    layers.append(nn.Linear(hidden_size, output_size))
    if layer_norm:
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)


class MeshGraphNet(nn.Module):
    """Encode-Process-Decode graph network regressing a node-wise output."""

    def __init__(self, node_in: int, edge_in: int, output_size: int = 3,
                 latent: int = 128, mlp_hidden: int = 128,
                 mlp_layers: int = 2, processor_steps: int = 10):
        super().__init__()
        self.node_encoder = build_mlp(node_in, latent, mlp_hidden, mlp_layers, layer_norm=True)
        self.edge_encoder = build_mlp(edge_in, latent, mlp_hidden, mlp_layers, layer_norm=True)
        self.edge_mlps = nn.ModuleList([
            build_mlp(3 * latent, latent, mlp_hidden, mlp_layers, layer_norm=True)
            for _ in range(processor_steps)])
        self.node_mlps = nn.ModuleList([
            build_mlp(2 * latent, latent, mlp_hidden, mlp_layers, layer_norm=True)
            for _ in range(processor_steps)])
        self.decoder = build_mlp(latent, output_size, mlp_hidden, mlp_layers, layer_norm=False)

    def forward(self, x, edge_attr, senders, receivers):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            e_in = torch.cat([edge_attr, x[senders], x[receivers]], dim=-1)
            edge_attr = edge_attr + edge_mlp(e_in)
            agg = torch.zeros_like(x)
            agg.index_add_(0, receivers, edge_attr)
            x = x + node_mlp(torch.cat([x, agg], dim=-1))
        return self.decoder(x)


def init_decoder_zero(model: MeshGraphNet) -> None:
    """Zero the decoder's final layer so the network starts at the trivial
    (do-nothing) predictor."""
    with torch.no_grad():
        last = model.decoder[-1]
        last.weight.zero_()
        last.bias.zero_()
