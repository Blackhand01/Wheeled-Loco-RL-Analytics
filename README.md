# Wheeled-Loco-RL-Analytics

JAX/MJX reinforcement learning stack for a wheeled Unitree Go1 quadruped.

## Components

- MuJoCo Menagerie Go1 assets adapted with four wheel joints and wheel actuators.
- JAX/MJX vectorized environment in `envs/wheeled_robot_env.py`.
- Flax actor-critic networks and PPO utilities in `algo/`.
- Training orchestration in `scripts/train.py`.
- Evaluation and golden dataset curation in `scripts/evaluate.py` and `data_management/`.

## Local Smoke Tests

```bash
source .venv/bin/activate
python scripts/train.py
python scripts/evaluate.py
```
