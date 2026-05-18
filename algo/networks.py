"""Flax actor-critic networks for PPO."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class ActorCritic(nn.Module):
    """Gaussian policy actor and scalar value critic in one Flax module."""

    action_dim: int = 16
    hidden_sizes: Sequence[int] = (256, 256, 256)
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        actor = obs
        for width in self.hidden_sizes:
            actor = nn.Dense(
                width,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(actor)
            actor = nn.tanh(actor)

        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(actor)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(self.log_std_init),
            (self.action_dim,),
        )
        log_std = jnp.broadcast_to(log_std, mean.shape)

        critic = obs
        for width in self.hidden_sizes:
            critic = nn.Dense(
                width,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=nn.initializers.zeros,
            )(critic)
            critic = nn.tanh(critic)

        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(critic)
        value = jnp.squeeze(value, axis=-1)

        return mean, log_std, value
