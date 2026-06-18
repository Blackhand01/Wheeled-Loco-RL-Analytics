# Wheeled-Loco-RL-Analytics

JAX/MJX reinforcement learning stack for a wheeled Unitree Go1 quadruped.

![Wheeled Go1 simulation preview](media/wheeled_go1_demo.gif)

This repository demonstrates an end-to-end robotics RL workflow: a wheel-legged
MuJoCo robot model, a JAX/MJX environment, PPO training utilities, evaluation
rollouts, and failure-case telemetry curation.

The preview above is a deterministic MuJoCo render of the modified robot model.
It is meant to show the morphology and simulation asset; trained policy behavior
is produced through the PPO/evaluation scripts below.

## What This Shows

- A Unitree Go1-style quadruped adapted with four wheel joints and wheel motors.
- A functional MJX environment with `reset`/`step`, observations, rewards and
  termination logic.
- A Flax actor-critic policy for continuous 16-dimensional actuator control.
- PPO in JAX with GAE, clipped policy loss, value clipping, entropy bonus,
  gradient clipping and minibatch epochs.
- Batched rollout collection using `jax.jit`, `jax.vmap` and `jax.lax.scan`.
- Evaluation telemetry for command tracking, stability, wheel speed, actuator
  forces and long-tail locomotion events.

## Repository Map

| Path | Purpose |
| --- | --- |
| `envs/assets/scene.xml` | MuJoCo scene for the wheeled Go1 model. |
| `envs/wheeled_robot_env.py` | JAX/MJX RL environment. |
| `algo/networks.py` | Flax Gaussian actor-critic network. |
| `algo/ppo_jax.py` | PPO loss, GAE and update utilities. |
| `scripts/train.py` | Local PPO training smoke run and full training entrypoint. |
| `scripts/evaluate.py` | Policy evaluation and telemetry export. |
| `scripts/render_demo.py` | Reproducible GIF/MP4/PNG renderer for the README media. |
| `data_management/curate_golden_set.py` | Slip, stall, near-fall and torque-saturation event curation. |

## Outputs

The evaluation script creates a local golden dataset at
`data_management/golden_dataset.pkl` with per-step telemetry and event summaries.
Generated datasets and checkpoints are intentionally ignored by Git.

Example smoke-test report:

```text
Evaluation report
  policy_source: random_init
  num_steps: 100
  stability_percent: 100.00
  TORQUE_SATURATION: 1
  STALL_EVENT: 1
```

Additional media:

- Static preview: `media/wheeled_go1_preview.png`
- MP4 preview: `media/wheeled_go1_demo.mp4`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run The Stack

```bash
python envs/wheeled_robot_env.py
python algo/ppo_jax.py
python scripts/train.py
python scripts/evaluate.py --steps 100
```

`scripts/train.py` defaults to a short local smoke run. Use `--full` to keep the
full values from `configs/config.yaml`.

```bash
python scripts/train.py --full
```

## Regenerate Media

```bash
python scripts/render_demo.py --output media/wheeled_go1_demo.gif --width 720 --height 405 --fps 20 --duration 4
python scripts/render_demo.py --output media/wheeled_go1_demo.mp4 --width 720 --height 408 --fps 20 --duration 4
python scripts/render_demo.py --output media/wheeled_go1_preview.png --width 1200 --height 675 --fps 20 --duration 2
```

MuJoCo/MJX may print optional `warp` import warnings on machines without Warp
installed. The CPU JAX path still runs the smoke tests.
