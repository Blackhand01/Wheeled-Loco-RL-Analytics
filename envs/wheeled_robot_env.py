"""JAX/MJX environment for the wheeled Unitree Go1 model."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class EnvState(NamedTuple):
    """Functional environment state.

    The state is a PyTree because both NamedTuple and ``mjx.Data`` are PyTrees,
    so it can be passed through ``jax.jit``, ``jax.vmap`` and ``jax.lax.scan``.
    """

    mjx_data: mjx.Data
    obs: jax.Array
    command: jax.Array
    rng: jax.Array
    steps: jax.Array


class WheeledLocoEnv:
    """Stateless JAX/MJX environment for the 16-actuator wheeled Go1."""

    def __init__(
        self,
        model_path: str | Path = "envs/assets/scene.xml",
        episode_length: int = 1000,
        min_torso_height: float = 0.18,
        min_up_z: float = 0.25,
        energy_weight: float = 1.0e-4,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        path = Path(model_path)
        if not path.is_absolute():
            path = root / path

        self.model_path = path
        self.mj_model = mujoco.MjModel.from_xml_path(str(path))
        self.mjx_model = mjx.put_model(self.mj_model)

        self.episode_length = jnp.asarray(episode_length, dtype=jnp.int32)
        self.min_torso_height = jnp.asarray(min_torso_height, dtype=jnp.float32)
        self.min_up_z = jnp.asarray(min_up_z, dtype=jnp.float32)
        self.energy_weight = jnp.asarray(energy_weight, dtype=jnp.float32)

        self.action_size = int(self.mj_model.nu)
        if self.action_size != 16:
            raise ValueError(f"Expected 16 actuators, got {self.action_size}.")

        home_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if home_id < 0:
            raise ValueError("Missing keyframe 'home' in MJCF model.")

        self.home_qpos = jnp.asarray(self.mj_model.key_qpos[home_id])
        self.home_qvel = jnp.asarray(self.mj_model.key_qvel[home_id])
        self.home_ctrl = jnp.asarray(self.mj_model.key_ctrl[home_id])
        self.ctrl_low = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_high = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 1])

        self.trunk_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.leg_qpos_idx = jnp.asarray(
            [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21],
            dtype=jnp.int32,
        )
        self.leg_qvel_idx = jnp.asarray(
            [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20],
            dtype=jnp.int32,
        )
        self.wheel_qvel_idx = jnp.asarray([9, 13, 17, 21], dtype=jnp.int32)

        self.obs_size = 38

    def reset(self, rng: jax.Array) -> EnvState:
        """Initializes the simulator at the 'home' keyframe."""
        rng, command_rng = jax.random.split(rng)
        command_low = jnp.asarray([-1.0, -0.5, -1.0], dtype=jnp.float32)
        command_high = jnp.asarray([2.0, 0.5, 1.0], dtype=jnp.float32)
        command = jax.random.uniform(
            command_rng,
            shape=(3,),
            minval=command_low,
            maxval=command_high,
        )

        data = mjx.make_data(self.mjx_model)
        data = data.replace(
            qpos=self.home_qpos,
            qvel=self.home_qvel,
            ctrl=self.home_ctrl,
        )
        data = mjx.forward(self.mjx_model, data)
        obs = self._get_obs(data, command)

        return EnvState(
            mjx_data=data,
            obs=obs,
            command=command,
            rng=rng,
            steps=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(
        self, state: EnvState, action: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        """Runs one MJX step with direct actuator controls."""
        action = jnp.asarray(action, dtype=jnp.float32)
        ctrl = jnp.clip(action, self.ctrl_low, self.ctrl_high)

        data = state.mjx_data.replace(ctrl=ctrl)
        data = mjx.step(self.mjx_model, data)
        obs = self._get_obs(data, state.command)

        reward = self._get_reward(data, state.command)
        terminated = self._is_terminated(data)
        steps = state.steps + jnp.asarray(1, dtype=jnp.int32)
        truncated = steps >= self.episode_length
        done = jnp.logical_or(terminated, truncated)

        new_state = EnvState(
            mjx_data=data,
            obs=obs,
            command=state.command,
            rng=state.rng,
            steps=steps,
        )
        return new_state, reward, done

    def _get_obs(self, data: mjx.Data, command: jax.Array) -> jax.Array:
        torso_quat = data.qpos[3:7]
        torso_gyro = data.qvel[3:6]
        leg_qpos = data.qpos[self.leg_qpos_idx]
        leg_qvel = data.qvel[self.leg_qvel_idx]
        wheel_qvel = data.qvel[self.wheel_qvel_idx]

        return jnp.concatenate(
            [torso_quat, torso_gyro, leg_qpos, leg_qvel, wheel_qvel, command]
        )

    def _get_reward(self, data: mjx.Data, command: jax.Array) -> jax.Array:
        linear_velocity = data.qvel[0:2]
        yaw_rate = data.qvel[5]

        linear_error = jnp.sum(jnp.square(linear_velocity - command[0:2]))
        angular_error = jnp.square(yaw_rate - command[2])
        tracking_reward = jnp.exp(-linear_error) + 0.5 * jnp.exp(-angular_error)

        torque_penalty = jnp.sum(jnp.square(data.actuator_force))
        return tracking_reward - self.energy_weight * torque_penalty

    def _is_terminated(self, data: mjx.Data) -> jax.Array:
        torso_height = data.xpos[self.trunk_body_id, 2]
        torso_up_z = self._quat_up_z(data.qpos[3:7])
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))

        too_low = torso_height < self.min_torso_height
        tipped_over = torso_up_z < self.min_up_z
        return jnp.logical_or(
            jnp.logical_or(too_low, tipped_over), jnp.logical_not(finite)
        )

    @staticmethod
    def _quat_up_z(quat: jax.Array) -> jax.Array:
        """Returns the world z component of the torso's local z axis."""
        _, x, y, _ = quat
        return 1.0 - 2.0 * (x * x + y * y)


if __name__ == "__main__":
    env = WheeledLocoEnv()
    rng = jax.random.PRNGKey(0)

    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    state = reset_jit(rng)
    rng, action_rng = jax.random.split(state.rng)
    action = jax.random.uniform(
        action_rng,
        shape=(env.action_size,),
        minval=env.ctrl_low,
        maxval=env.ctrl_high,
    )
    next_state, reward, done = step_jit(state, action)

    print("obs shape:", next_state.obs.shape)
    print("reward:", float(reward))
    print("done:", bool(done))
