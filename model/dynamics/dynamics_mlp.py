"""
MLP models for diffusion policies.

"""

import torch
import torch.nn as nn
import logging
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)


class DynamicsMLP(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        horizon_steps,
        cond_steps,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        device="cpu",
        checkpoint_path=None,
    ):
        super().__init__()

        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        input_dim = obs_dim * cond_steps + action_dim * horizon_steps
        output_dim = obs_dim
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.to(device)
        if checkpoint_path is not None:
            print(f"Loading dynamics model from {checkpoint_path}")
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(
        self,
        state,
        action,
        **kwargs,
    ):
        """
        state: (B, To, Do)
        action: (B, To, Da)
        """
        B, To, Do = state.shape
        B, Ta, Da = action.shape

        x = torch.cat([state.view(B, -1), action.view(B, -1)], dim=-1)
        out = self.mlp_mean(x)
        return out.view(B, Do)
    
    def loss(self, states, actions, end_state):
        pred = self.forward(states, actions)
        loss = torch.nn.functional.mse_loss(pred, end_state)
        return loss