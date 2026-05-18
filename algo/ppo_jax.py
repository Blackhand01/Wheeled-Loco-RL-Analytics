"""Functional PPO utilities in JAX/Flax/Optax."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from algo.networks import ActorCritic


Array = jax.Array
Params = Any
ApplyFn = Callable[[Params, Array], tuple[Array, Array, Array]]


class PPOBatch(NamedTuple):
    obs: Array
    actions: Array
    log_probs: Array
    values: Array
    advantages: Array
    returns: Array


def gaussian_log_prob(actions: Array, mean: Array, log_std: Array) -> Array:
    """Returns summed log-probability for a diagonal Gaussian policy."""
    log_two_pi = jnp.log(2.0 * jnp.pi)
    inv_var = jnp.exp(-2.0 * log_std)
    log_prob = -0.5 * (
        jnp.square(actions - mean) * inv_var + 2.0 * log_std + log_two_pi
    )
    return jnp.sum(log_prob, axis=-1)


def gaussian_entropy(log_std: Array) -> Array:
    """Returns summed entropy for a diagonal Gaussian policy."""
    entropy = log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    return jnp.sum(entropy, axis=-1)


def create_optimizer(
    learning_rate: float = 3.0e-4,
    max_grad_norm: float = 0.5,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


def create_train_state(
    rng: Array,
    model: ActorCritic,
    sample_obs: Array,
    tx: optax.GradientTransformation,
) -> TrainState:
    params = model.init(rng, sample_obs)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def ppo_loss(
    params: Params,
    apply_fn: ApplyFn,
    batch: PPOBatch,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[Array, dict[str, Array]]:
    mean, log_std, values = apply_fn(params, batch.obs)
    new_log_probs = gaussian_log_prob(batch.actions, mean, log_std)
    entropy = gaussian_entropy(log_std)

    advantages = (batch.advantages - jnp.mean(batch.advantages)) / (
        jnp.std(batch.advantages) + 1.0e-8
    )
    ratio = jnp.exp(new_log_probs - batch.log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -jnp.mean(
        jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    )

    value_pred_clipped = batch.values + jnp.clip(
        values - batch.values, -clip_ratio, clip_ratio
    )
    value_losses = jnp.square(values - batch.returns)
    value_losses_clipped = jnp.square(value_pred_clipped - batch.returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

    entropy_loss = jnp.mean(entropy)
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

    approx_kl = jnp.mean(batch.log_probs - new_log_probs)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_ratio).astype(jnp.float32))

    metrics = {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
    return total_loss, metrics


def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    last_value: Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Array, Array]:
    """Computes GAE advantages and returns over leading time dimension.

    Expected shapes are ``[T]`` or ``[T, num_envs]`` for rewards, values and
    dones, with ``last_value`` matching the non-time dimensions.
    """
    next_values = jnp.concatenate([values[1:], last_value[jnp.newaxis, ...]], axis=0)

    def scan_fn(next_advantage: Array, transition: tuple[Array, Array, Array, Array]):
        reward, value, done, next_value = transition
        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * not_done * next_value - value
        advantage = delta + gamma * gae_lambda * not_done * next_advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(last_value),
        (rewards, values, dones, next_values),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def _flatten_rollout(batch: PPOBatch) -> PPOBatch:
    """Flattens [time, env, ...] rollout tensors to [batch, ...] tensors."""
    if batch.obs.ndim <= 2:
        return batch
    return PPOBatch(
        obs=batch.obs.reshape((-1,) + batch.obs.shape[2:]),
        actions=batch.actions.reshape((-1,) + batch.actions.shape[2:]),
        log_probs=batch.log_probs.reshape((-1,)),
        values=batch.values.reshape((-1,)),
        advantages=batch.advantages.reshape((-1,)),
        returns=batch.returns.reshape((-1,)),
    )


def update_minibatch(
    train_state: TrainState,
    batch: PPOBatch,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[TrainState, dict[str, Array]]:
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (_, metrics), grads = grad_fn(
        train_state.params,
        train_state.apply_fn,
        batch,
        clip_ratio,
        vf_coef,
        ent_coef,
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, metrics


@partial(jax.jit, static_argnames=("num_minibatches",))
def update_epoch(
    train_state: TrainState,
    batch: PPOBatch,
    rng: Array,
    num_minibatches: int = 4,
    clip_ratio: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[TrainState, dict[str, Array]]:
    """Shuffles a rollout, splits it into minibatches, and applies PPO updates."""
    batch = _flatten_rollout(batch)
    batch_size = batch.obs.shape[0]
    if batch_size % num_minibatches != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by {num_minibatches}."
        )

    permutation = jax.random.permutation(rng, batch_size)
    shuffled = jax.tree_util.tree_map(lambda x: x[permutation], batch)
    minibatch_size = batch_size // num_minibatches
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape((num_minibatches, minibatch_size) + x.shape[1:]),
        shuffled,
    )

    def scan_fn(state: TrainState, minibatch: PPOBatch):
        return update_minibatch(state, minibatch, clip_ratio, vf_coef, ent_coef)

    return jax.lax.scan(scan_fn, train_state, minibatches)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    obs_dim = 38
    action_dim = 16
    batch_size = 64

    model = ActorCritic(action_dim=action_dim)
    tx = create_optimizer()

    rng, init_rng, data_rng, action_rng, update_rng = jax.random.split(rng, 5)
    sample_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    train_state = create_train_state(init_rng, model, sample_obs, tx)

    obs = jax.random.normal(data_rng, (batch_size, obs_dim))
    mean, log_std, values = train_state.apply_fn(train_state.params, obs)
    actions = mean + jnp.exp(log_std) * jax.random.normal(
        action_rng, (batch_size, action_dim)
    )
    log_probs = gaussian_log_prob(actions, mean, log_std)
    advantages = jax.random.normal(update_rng, (batch_size,))
    returns = values + advantages

    batch = PPOBatch(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        advantages=advantages,
        returns=returns,
    )

    train_state, metrics = update_epoch(
        train_state,
        batch,
        update_rng,
        num_minibatches=4,
    )
    jax.block_until_ready(metrics["loss"])

    print("updated step:", int(train_state.step))
    print("loss shape:", metrics["loss"].shape)
    print("last loss:", float(metrics["loss"][-1]))
