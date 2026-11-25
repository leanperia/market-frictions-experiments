"""EIIE feature extractor for SB3 policies.

EIIE — *Ensemble of Identical Independent Evaluators* — from Jiang,
Xu & Liang (2017). The defining property is **weight sharing across
the asset axis**: the same small network is applied identically to
each stock's feature vector, then the per-asset embeddings are
aggregated for the policy/value heads.

This extractor takes the flat observation that
``PortfolioOptimizationImpactEnv`` produces:

    [ price_var(N), current_weights(1+N), last_action(1+N),
      perm_impact(N), tech(N * tech_features_per_asset) ]

and reshapes the per-asset slices into a tensor of shape
``(batch, channels_per_asset, N)``, then applies a 1×1 Conv1d
(equivalent to a shared MLP scanned across the asset axis). The two
cash-bucket scalars (cash weight + cash last-action) are appended
separately, since they are not "per asset". The output feature dim is
``hidden_dim * N + 2``; SB3 adds the standard MLP head on top.

Reference
---------
Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning
Framework for the Financial Portfolio Management Problem. arXiv:1706.10059.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EIIEExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_assets: int,
        tech_features_per_asset: int,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("EIIEExtractor expects a Box observation space.")
        n = int(n_assets)
        t = int(tech_features_per_asset)
        expected = n + (1 + n) + (1 + n) + n + n * t
        actual = int(observation_space.shape[0])
        if actual != expected:
            raise ValueError(
                f"Observation length {actual} does not match the layout for n_assets={n}, "
                f"tech_features_per_asset={t}: expected {expected}."
            )

        per_asset_channels = 4 + t  # price_var, weight, last_action, perm_impact, + tech features
        features_dim = hidden_dim * n + 2  # +2 scalars for cash bucket
        super().__init__(observation_space, features_dim=features_dim)

        self.n_assets = n
        self.tech_per_asset = t
        self.per_asset_channels = per_asset_channels
        self.hidden_dim = hidden_dim

        layers = [
            nn.Conv1d(per_asset_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        ])
        self.shared = nn.Sequential(*layers)

    def _slice(self, obs: torch.Tensor):
        n = self.n_assets
        price_var = obs[:, 0:n]                  # (B, N)
        cash_weight = obs[:, n : n + 1]          # (B, 1)
        stock_weights = obs[:, n + 1 : 2 * n + 1]
        cash_last = obs[:, 2 * n + 1 : 2 * n + 2]
        stock_last = obs[:, 2 * n + 2 : 3 * n + 2]
        perm = obs[:, 3 * n + 2 : 4 * n + 2]
        tech_start = 4 * n + 2
        tech = obs[:, tech_start:]               # (B, N*tech)
        tech = tech.view(obs.shape[0], n, self.tech_per_asset)  # (B, N, tech)
        return price_var, cash_weight, stock_weights, cash_last, stock_last, perm, tech

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() != 2:
            observations = observations.view(observations.shape[0], -1)
        price_var, cash_w, stock_w, cash_l, stock_l, perm, tech = self._slice(observations)

        # (B, channels=4 + tech_per_asset, N)
        per_asset = torch.stack(
            [price_var, stock_w, stock_l, perm], dim=1
        )  # (B, 4, N)
        tech_perasset = tech.permute(0, 2, 1)  # (B, tech, N)
        x = torch.cat([per_asset, tech_perasset], dim=1)  # (B, 4+tech, N)

        embedded = self.shared(x)              # (B, hidden, N)
        flat = embedded.view(embedded.shape[0], -1)  # (B, hidden*N)
        return torch.cat([flat, cash_w, cash_l], dim=1)  # (B, hidden*N + 2)
