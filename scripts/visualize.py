import os
import sys
import time
import argparse
import pickle
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from flax.serialization import from_bytes

# Add the project root to the import path for local modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.wheeled_robot_env import WheeledLocoEnv
from algo.networks import ActorCritic


def build_obs(data, env, command):
    torso_quat = jnp.asarray(data.qpos[3:7])
    torso_gyro = jnp.asarray(data.qvel[3:6])
    leg_qpos = jnp.asarray(data.qpos[np.asarray(env.leg_qpos_idx)])
    leg_qvel = jnp.asarray(data.qvel[np.asarray(env.leg_qvel_idx)])
    wheel_qvel = jnp.asarray(data.qvel[np.asarray(env.wheel_qvel_idx)])
    return jnp.concatenate(
        [torso_quat, torso_gyro, leg_qpos, leg_qvel, wheel_qvel, command]
    )[None, :]


def load_checkpoint(path, params):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    candidate = payload.get("params", payload) if isinstance(payload, dict) else payload
    if isinstance(candidate, (bytes, bytearray)):
        return from_bytes(params, candidate)
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the robot policy in the MuJoCo Viewer."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a policy checkpoint (.pkl).",
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Use the policy even without a checkpoint. By default the robot holds the home pose.",
    )
    args = parser.parse_args()

    print("Initializing visual environment...")
    # Use standard MuJoCo for local interactive rendering.
    model = mujoco.MjModel.from_xml_path("envs/assets/scene.xml")
    data = mujoco.MjData(model)

    # Initialize the JAX environment to reuse the observation index mapping.
    env = WheeledLocoEnv()

    # Policy setup.
    init_obs = jnp.zeros((1, 38))
    policy = ActorCritic(action_dim=16)
    rng = jax.random.PRNGKey(0)
    policy_params = policy.init(rng, init_obs)
    apply_policy = jax.jit(lambda params, obs: policy.apply(params, obs)[0])

    use_policy = args.policy
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        policy_params = load_checkpoint(args.checkpoint, policy_params)
        use_policy = True
    else:
        print("No checkpoint found or specified.")
        if use_policy:
            print("Running with a random untrained policy: the robot will probably fall.")
        else:
            print("Home mode: holding keyframe targets for stable visual inspection.")

    print("\nOpening MuJoCo Viewer... Press ESC in the window to close it.")

    # Launch MuJoCo's native viewer on macOS.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the robot to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, 0)

        # Target command used by the policy observation.
        command = jnp.array([1.0, 0.0, 0.0])

        # Real-time simulation loop.
        while viewer.is_running():
            step_start = time.time()

            if use_policy:
                # 1. Extract the current observation directly from mujoco.Data.
                obs = build_obs(data, env, command)

                # 2. Compute the action with the Flax policy.
                actions = np.asarray(apply_policy(policy_params, obs)[0])
                actions = np.clip(
                    actions,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1],
                )
            else:
                actions = model.key_ctrl[0].copy()

            # 3. Apply the actions to the robot actuators.
            data.ctrl[:] = actions

            # 4. Step physics.
            mujoco.mj_step(model, data)

            # 5. Sync the graphical viewer.
            viewer.sync()

            # Keep the loop close to real time.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
