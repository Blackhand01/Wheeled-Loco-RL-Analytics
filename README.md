# Wheeled-Loco-RL-Analytics

JAX/MJX reinforcement learning stack for a wheeled Unitree Go1 quadruped.

This repository is scoped as a robotics RL portfolio project: it combines a
wheel-legged MuJoCo model, a vectorized JAX/MJX environment, PPO training, and
evaluation telemetry for long-tail locomotion events.

## Components

- MuJoCo Menagerie Go1 assets adapted with four wheel joints and wheel actuators.
- JAX/MJX vectorized environment in `envs/wheeled_robot_env.py`.
- Flax actor-critic networks and PPO utilities in `algo/`.
- Training orchestration in `scripts/train.py`.
- Evaluation and golden dataset curation in `scripts/evaluate.py` and `data_management/`.

## Relevant Capabilities

- Implements a diagonal-Gaussian actor-critic policy in Flax.
- Uses clipped PPO updates with GAE, entropy regularization, value clipping and
  minibatch epochs.
- Runs batched rollouts with `jax.vmap`, `jax.jit` and `jax.lax.scan`.
- Tracks command-following reward, energy penalty, fall termination and episode
  returns.
- Curates evaluation telemetry into slip, near-fall, torque-saturation and stall
  event categories.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Local Smoke Tests

```bash
source .venv/bin/activate
python scripts/train.py
python scripts/evaluate.py --steps 100
```
