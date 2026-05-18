"""Main PPO training orchestrator for the wheeled Go1 MJX environment."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import jax
import jax.numpy as jnp
import optax
import yaml
from flax.serialization import to_bytes
from flax.training.train_state import TrainState

from algo.networks import ActorCritic
from algo.ppo_jax import (
    PPOBatch,
    compute_gae,
    gaussian_log_prob,
    update_epoch,
)
from envs.wheeled_robot_env import EnvState, WheeledLocoEnv


class RolloutBatch(NamedTuple):
    obs: jax.Array
    actions: jax.Array
    log_probs: jax.Array
    rewards: jax.Array
    dones: jax.Array
    values: jax.Array
    episode_returns: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_states: EnvState
    rng: jax.Array
    episode_returns: jax.Array


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def init_wandb(config: dict[str, Any]):
    mode = config["logging"].get("wandb_mode", "disabled")
    try:
        import wandb
    except ImportError:
        return None

    if mode in ("disabled", "offline", "online"):
        return wandb.init(
            project=config["logging"]["wandb_project"],
            config=config,
            mode=mode,
        )
    return wandb.init(
        project=config["logging"]["wandb_project"],
        config=config,
        mode="disabled",
    )


def make_train_state(
    rng: jax.Array,
    env: WheeledLocoEnv,
    config: dict[str, Any],
) -> TrainState:
    model = ActorCritic(action_dim=env.action_size)
    sample_obs = jnp.zeros((1, env.obs_size), dtype=jnp.float32)
    params = model.init(rng, sample_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["ppo"]["max_grad_norm"]),
        optax.adam(config["ppo"]["lr"]),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def tree_where(mask: jax.Array, true_tree: Any, false_tree: Any) -> Any:
    def choose(true_leaf, false_leaf):
        shaped_mask = mask.reshape(mask.shape + (1,) * (true_leaf.ndim - mask.ndim))
        return jnp.where(shaped_mask, true_leaf, false_leaf)

    return jax.tree_util.tree_map(choose, true_tree, false_tree)


def make_collect_rollouts(env: WheeledLocoEnv, num_envs: int, num_steps: int):
    reset_vmap = jax.vmap(env.reset)
    step_vmap = jax.vmap(env.step)

    @jax.jit
    def collect_rollouts(runner_state: RunnerState) -> tuple[RunnerState, RolloutBatch]:
        def rollout_step(carry: RunnerState, _: None):
            train_state, env_states, rng, running_returns = carry
            rng, action_rng, reset_rng = jax.random.split(rng, 3)

            mean, log_std, values = train_state.apply_fn(
                train_state.params, env_states.obs
            )
            noise = jax.random.normal(action_rng, mean.shape)
            actions = mean + jnp.exp(log_std) * noise
            log_probs = gaussian_log_prob(actions, mean, log_std)

            next_env_states, rewards, dones = step_vmap(env_states, actions)
            updated_returns = running_returns + rewards
            completed_returns = jnp.where(dones, updated_returns, 0.0)
            next_running_returns = jnp.where(dones, 0.0, updated_returns)

            reset_keys = jax.random.split(reset_rng, num_envs)
            reset_env_states = reset_vmap(reset_keys)
            next_env_states = tree_where(dones, reset_env_states, next_env_states)

            transition = RolloutBatch(
                obs=env_states.obs,
                actions=actions,
                log_probs=log_probs,
                rewards=rewards,
                dones=dones,
                values=values,
                episode_returns=completed_returns,
            )
            return (
                RunnerState(train_state, next_env_states, rng, next_running_returns),
                transition,
            )

        return jax.lax.scan(rollout_step, runner_state, None, length=num_steps)

    return collect_rollouts


@jax.jit
def value_from_state(train_state: TrainState, env_states: EnvState) -> jax.Array:
    _, _, values = train_state.apply_fn(train_state.params, env_states.obs)
    return values


def make_ppo_batch(
    rollout: RolloutBatch,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> PPOBatch:
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return PPOBatch(
        obs=rollout.obs,
        actions=rollout.actions,
        log_probs=rollout.log_probs,
        values=rollout.values,
        advantages=advantages,
        returns=returns,
    )


def save_checkpoint(train_state: TrainState, checkpoint_dir: str | Path, step: int) -> Path:
    path = Path(checkpoint_dir)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / f"ppo_step_{step:010d}.pkl"
    payload = {
        "step": int(step),
        "params": to_bytes(train_state.params),
    }
    with ckpt_path.open("wb") as f:
        pickle.dump(payload, f)
    return ckpt_path


def run_training(config: dict[str, Any]) -> TrainState:
    wandb_run = init_wandb(config)

    rng = jax.random.PRNGKey(config["seed"])
    rng, init_rng, reset_rng = jax.random.split(rng, 3)

    env = WheeledLocoEnv(episode_length=config["env"]["episode_length"])
    num_envs = int(config["env"]["num_envs"])
    num_steps = int(config["ppo"]["num_steps"])
    rollout_size = num_envs * num_steps
    minibatch_size = int(config["ppo"]["batch_size"])
    if rollout_size % minibatch_size != 0:
        raise ValueError(
            f"num_envs * num_steps ({rollout_size}) must be divisible by "
            f"batch_size ({minibatch_size})."
        )
    num_minibatches = rollout_size // minibatch_size
    num_iterations = max(1, int(config["ppo"]["total_timesteps"]) // rollout_size)

    train_state = make_train_state(init_rng, env, config)
    reset_keys = jax.random.split(reset_rng, num_envs)
    env_states = jax.jit(jax.vmap(env.reset))(reset_keys)
    runner_state = RunnerState(train_state, env_states, rng, jnp.zeros((num_envs,)))
    collect_rollouts = make_collect_rollouts(env, num_envs, num_steps)

    for iteration in range(1, num_iterations + 1):
        runner_state, rollout = collect_rollouts(runner_state)
        last_value = value_from_state(runner_state.train_state, runner_state.env_states)
        ppo_batch = make_ppo_batch(
            rollout,
            last_value,
            gamma=config["ppo"]["gamma"],
            gae_lambda=config["ppo"]["gae_lambda"],
        )

        train_state = runner_state.train_state
        update_metrics = None
        rng = runner_state.rng
        for _ in range(int(config["ppo"]["num_epochs"])):
            rng, update_rng = jax.random.split(rng)
            train_state, update_metrics = update_epoch(
                train_state,
                ppo_batch,
                update_rng,
                num_minibatches=num_minibatches,
                clip_ratio=config["ppo"]["clip_ratio"],
                vf_coef=config["ppo"]["value_coef"],
                ent_coef=config["ppo"]["entropy_coef"],
            )
        runner_state = RunnerState(
            train_state,
            runner_state.env_states,
            rng,
            runner_state.episode_returns,
        )

        mean_reward = float(jnp.mean(rollout.rewards))
        completed_count = jnp.sum(rollout.dones.astype(jnp.float32))
        episode_return_mean = float(
            jnp.where(
                completed_count > 0.0,
                jnp.sum(rollout.episode_returns) / jnp.maximum(completed_count, 1.0),
                mean_reward,
            )
        )
        done_rate = float(jnp.mean(rollout.dones.astype(jnp.float32)))
        metrics = {
            "iteration": iteration,
            "timesteps": iteration * rollout_size,
            "reward_mean": mean_reward,
            "episode_return_mean": episode_return_mean,
            "done_rate": done_rate,
        }
        if update_metrics is not None:
            metrics.update(
                {
                    key: float(jnp.mean(value))
                    for key, value in update_metrics.items()
                }
            )

        print(
            "iter={iteration} steps={timesteps} reward={reward_mean:.4f} "
            "episode_return={episode_return_mean:.4f} "
            "policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
            "entropy={entropy:.4f}".format(**metrics)
        )
        if wandb_run is not None:
            wandb_run.log(metrics, step=metrics["timesteps"])

        save_interval = int(config["logging"]["save_interval"])
        if save_interval > 0 and iteration % save_interval == 0:
            save_checkpoint(
                runner_state.train_state,
                config["logging"]["checkpoint_dir"],
                metrics["timesteps"],
            )

    if wandb_run is not None:
        wandb_run.finish()
    return runner_state.train_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the config as-is instead of the default local smoke test.",
    )
    parser.add_argument("--test", action="store_true", help="Alias for the default smoke test.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.test or not args.full:
        config = deep_update(
            config,
            {
                "env": {"num_envs": 4, "episode_length": 1000},
                "ppo": {
                    "total_timesteps": 512,
                    "num_steps": 32,
                    "batch_size": 64,
                    "num_epochs": 1,
                },
                "logging": {"wandb_mode": "disabled", "save_interval": 0},
            },
        )
    run_training(config)


if __name__ == "__main__":
    main()
