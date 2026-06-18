"""Policy evaluation and golden dataset curation."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes

from algo.networks import ActorCritic
from data_management.curate_golden_set import GoldenDatasetCurator, quaternion_to_euler
from envs.wheeled_robot_env import EnvState, WheeledLocoEnv


def load_policy_params(
    checkpoint: str | Path | None,
    model: ActorCritic,
    sample_obs: jax.Array,
    rng: jax.Array,
) -> tuple[Any, str]:
    params = model.init(rng, sample_obs)
    if checkpoint is None:
        return params, "random_init"

    path = Path(checkpoint)
    if not path.exists():
        return params, f"random_init_missing_checkpoint:{path}"

    with path.open("rb") as f:
        payload = pickle.load(f)

    candidate = payload.get("params", payload) if isinstance(payload, dict) else payload
    if isinstance(candidate, (bytes, bytearray)):
        params = from_bytes(params, candidate)
    else:
        params = candidate
    return params, f"checkpoint:{path}"


def command_schedule(num_steps: int) -> np.ndarray:
    """Stress-test commands with accelerations, reversals and yaw turns."""
    anchors = np.asarray(
        [
            [1.6, 0.0, 0.0],
            [2.0, 0.0, 0.8],
            [-1.2, 0.0, 0.0],
            [0.0, 0.8, -1.0],
            [1.4, -0.5, 1.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    segment = max(1, int(np.ceil(num_steps / anchors.shape[0])))
    return np.repeat(anchors, segment, axis=0)[:num_steps]


def make_eval_step(env: WheeledLocoEnv, model: ActorCritic):
    @jax.jit
    def eval_step(params: Any, state: EnvState, command: jax.Array):
        state = state._replace(
            command=command,
            obs=env._get_obs(state.mjx_data, command),
        )
        mean, _, value = model.apply(params, state.obs)
        next_state, reward, done = env.step(state, mean)
        return next_state, mean, reward, done, value

    return eval_step


def append_telemetry(
    telemetry: dict[str, list[Any]],
    env: WheeledLocoEnv,
    state: EnvState,
    action: jax.Array,
    command: np.ndarray,
    reward: jax.Array,
    done: jax.Array,
    value: jax.Array,
    step: int,
) -> None:
    data = state.mjx_data
    quat = np.asarray(data.qpos[3:7])
    roll, pitch, yaw = quaternion_to_euler(quat[None, :])

    telemetry["steps"].append(step)
    telemetry["torso_quat"].append(quat)
    telemetry["roll"].append(float(roll[0]))
    telemetry["pitch"].append(float(pitch[0]))
    telemetry["yaw"].append(float(yaw[0]))
    telemetry["torso_linear_velocity"].append(np.asarray(data.qvel[0:3]))
    telemetry["torso_angular_velocity"].append(np.asarray(data.qvel[3:6]))
    telemetry["wheel_angular_velocity"].append(np.asarray(data.qvel[env.wheel_qvel_idx]))
    telemetry["actuator_force"].append(np.asarray(data.actuator_force))
    telemetry["actions"].append(np.asarray(action))
    telemetry["commands"].append(command)
    telemetry["rewards"].append(float(reward))
    telemetry["dones"].append(bool(done))
    telemetry["values"].append(float(value))
    telemetry["torso_height"].append(float(data.xpos[env.trunk_body_id, 2]))


def finalize_telemetry(
    telemetry: dict[str, list[Any]],
    env: WheeledLocoEnv,
) -> dict[str, Any]:
    out = {key: np.asarray(value) for key, value in telemetry.items()}

    force_limited = np.asarray(env.mj_model.actuator_forcelimited, dtype=bool)
    force_ranges = np.asarray(env.mj_model.actuator_forcerange)
    torque_limits = np.full((env.action_size,), np.nan, dtype=np.float64)
    if force_ranges.shape[0] == env.action_size:
        torque_limits[force_limited] = np.max(np.abs(force_ranges[force_limited]), axis=-1)

    ctrl_low = np.asarray(env.ctrl_low)
    ctrl_high = np.asarray(env.ctrl_high)
    out["torque_limits"] = torque_limits
    out["action_limits"] = np.maximum(np.abs(ctrl_low), np.abs(ctrl_high))
    return out


def evaluate_policy(
    checkpoint: str | Path | None = None,
    num_steps: int = 500,
    output_path: str | Path = ROOT / "data_management/golden_dataset.pkl",
    seed: int = 0,
) -> dict[str, Any]:
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, reset_rng = jax.random.split(rng, 3)

    env = WheeledLocoEnv(episode_length=max(num_steps + 1, 1000))
    model = ActorCritic(action_dim=env.action_size)
    sample_obs = jnp.zeros((env.obs_size,), dtype=jnp.float32)
    params, source = load_policy_params(checkpoint, model, sample_obs, init_rng)

    reset = jax.jit(env.reset)
    eval_step = make_eval_step(env, model)
    state = reset(reset_rng)
    commands = command_schedule(num_steps)

    telemetry: dict[str, list[Any]] = {
        "steps": [],
        "torso_quat": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
        "torso_linear_velocity": [],
        "torso_angular_velocity": [],
        "wheel_angular_velocity": [],
        "actuator_force": [],
        "actions": [],
        "commands": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "torso_height": [],
    }

    for step, command in enumerate(commands):
        state, action, reward, done, value = eval_step(
            params, state, jnp.asarray(command, dtype=jnp.float32)
        )
        append_telemetry(telemetry, env, state, action, command, reward, done, value, step)
        if bool(done):
            rng, reset_key = jax.random.split(rng)
            state = reset(reset_key)

    episode_telemetry = finalize_telemetry(telemetry, env)
    curator = GoldenDatasetCurator()
    curator.add_episode(episode_telemetry, episode_id="evaluation_0")
    saved_path = curator.save_dataset(output_path)

    report = curator.report()
    report.update(
        {
            "policy_source": source,
            "num_steps": int(num_steps),
            "reward_mean": float(np.mean(episode_telemetry["rewards"])),
            "reward_variance": float(np.var(episode_telemetry["rewards"])),
            "dataset_path": str(saved_path),
        }
    )
    return report


def print_report(report: dict[str, Any]) -> None:
    print("Evaluation report")
    print(f"  policy_source: {report['policy_source']}")
    print(f"  num_steps: {report['num_steps']}")
    print(f"  reward_mean: {report['reward_mean']:.4f}")
    print(f"  reward_variance: {report['reward_variance']:.4f}")
    print(f"  stability_percent: {report['stability_percent']:.2f}")
    print(f"  num_events: {report['num_events']}")
    for event_type, count in report["event_counts"].items():
        print(f"  {event_type}: {count}")
    print(f"  dataset_path: {report['dataset_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--output",
        default=str(ROOT / "data_management/golden_dataset.pkl"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = evaluate_policy(
        checkpoint=args.checkpoint,
        num_steps=args.steps,
        output_path=args.output,
        seed=args.seed,
    )
    print_report(report)


if __name__ == "__main__":
    main()
