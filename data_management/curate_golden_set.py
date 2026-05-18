"""Golden dataset curation for wheeled quadruped long-tail events."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EDGE_CASE_TAXONOMY = {
    "SLIP_EVENT": {
        "description": "Wheel angular speed is high while torso planar speed is low.",
        "signals": ["wheel_angular_velocity", "torso_linear_velocity"],
    },
    "NEAR_FALL_EVENT": {
        "description": "Roll or pitch exceeds 45 degrees without immediate termination.",
        "signals": ["roll", "pitch", "done"],
    },
    "TORQUE_SATURATION": {
        "description": "Actuators remain near available torque/control limits for multiple steps.",
        "signals": ["actuator_force", "action", "torque_limits", "action_limits"],
    },
    "STALL_EVENT": {
        "description": "Commanded planar speed is high while torso planar speed remains low.",
        "signals": ["command", "torso_linear_velocity"],
    },
}


@dataclass(frozen=True)
class CurationThresholds:
    slip_wheel_speed: float = 8.0
    slip_torso_speed: float = 0.15
    near_fall_angle: float = 0.785
    torque_saturation_fraction: float = 0.95
    torque_saturation_steps: int = 8
    stall_command_speed: float = 0.8
    stall_torso_speed: float = 0.12
    stall_min_steps: int = 8


def quaternion_to_euler(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts MuJoCo quaternions [w, x, y, z] to roll, pitch, yaw arrays."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat.T

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _as_array(telemetry: dict[str, Any], key: str, default: Any = None) -> np.ndarray:
    if key in telemetry:
        return np.asarray(telemetry[key])
    if default is None:
        raise KeyError(f"Telemetry is missing required field '{key}'.")
    return np.asarray(default)


def _contiguous_regions(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    regions = list(zip(changes[::2], changes[1::2], strict=True))
    return [(start, end) for start, end in regions if end - start >= min_length]


def _severity(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values) / max(threshold, 1.0e-6))


def categorize_episode(
    telemetry: dict[str, Any],
    thresholds: CurationThresholds | None = None,
    episode_id: str = "episode_0",
) -> list[dict[str, Any]]:
    """Applies the edge-case taxonomy to one episode of telemetry."""
    thresholds = thresholds or CurationThresholds()

    quat = _as_array(telemetry, "torso_quat")
    roll = np.asarray(telemetry.get("roll", quaternion_to_euler(quat)[0]))
    pitch = np.asarray(telemetry.get("pitch", quaternion_to_euler(quat)[1]))
    torso_velocity = _as_array(telemetry, "torso_linear_velocity")
    wheel_velocity = _as_array(telemetry, "wheel_angular_velocity")
    actions = _as_array(telemetry, "actions")
    actuator_force = _as_array(telemetry, "actuator_force", np.zeros_like(actions))
    commands = _as_array(telemetry, "commands")
    dones = _as_array(telemetry, "dones", np.zeros((quat.shape[0],), dtype=bool)).astype(bool)
    rewards = _as_array(telemetry, "rewards", np.zeros((quat.shape[0],)))

    planar_speed = np.linalg.norm(torso_velocity[:, :2], axis=-1)
    wheel_speed = np.mean(np.abs(wheel_velocity), axis=-1)
    command_speed = np.linalg.norm(commands[:, :2], axis=-1)

    events: list[dict[str, Any]] = []

    slip_mask = (
        (wheel_speed > thresholds.slip_wheel_speed)
        & (planar_speed < thresholds.slip_torso_speed)
        & ~dones
    )
    for start, end in _contiguous_regions(slip_mask):
        events.append(
            _make_event(
                episode_id,
                "SLIP_EVENT",
                start,
                end,
                severity=_severity(wheel_speed[start:end], thresholds.slip_wheel_speed),
                metrics={
                    "mean_wheel_speed": float(np.mean(wheel_speed[start:end])),
                    "mean_planar_speed": float(np.mean(planar_speed[start:end])),
                    "mean_reward": float(np.mean(rewards[start:end])),
                },
            )
        )

    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    near_fall_mask = (tilt > thresholds.near_fall_angle) & ~dones
    for start, end in _contiguous_regions(near_fall_mask):
        events.append(
            _make_event(
                episode_id,
                "NEAR_FALL_EVENT",
                start,
                end,
                severity=_severity(tilt[start:end], thresholds.near_fall_angle),
                metrics={
                    "max_roll_rad": float(np.max(np.abs(roll[start:end]))),
                    "max_pitch_rad": float(np.max(np.abs(pitch[start:end]))),
                    "mean_reward": float(np.mean(rewards[start:end])),
                },
            )
        )

    saturation_mask = _torque_saturation_mask(
        telemetry,
        actions,
        actuator_force,
        thresholds.torque_saturation_fraction,
    )
    for start, end in _contiguous_regions(
        saturation_mask, min_length=thresholds.torque_saturation_steps
    ):
        events.append(
            _make_event(
                episode_id,
                "TORQUE_SATURATION",
                start,
                end,
                severity=float(np.mean(saturation_mask[start:end])),
                metrics={
                    "mean_abs_action": float(np.mean(np.abs(actions[start:end]))),
                    "mean_abs_actuator_force": float(np.mean(np.abs(actuator_force[start:end]))),
                    "mean_reward": float(np.mean(rewards[start:end])),
                },
            )
        )

    stall_mask = (
        (command_speed > thresholds.stall_command_speed)
        & (planar_speed < thresholds.stall_torso_speed)
        & ~dones
    )
    for start, end in _contiguous_regions(stall_mask, min_length=thresholds.stall_min_steps):
        events.append(
            _make_event(
                episode_id,
                "STALL_EVENT",
                start,
                end,
                severity=_severity(command_speed[start:end], thresholds.stall_command_speed),
                metrics={
                    "mean_command_speed": float(np.mean(command_speed[start:end])),
                    "mean_planar_speed": float(np.mean(planar_speed[start:end])),
                    "mean_reward": float(np.mean(rewards[start:end])),
                },
            )
        )

    return events


def _torque_saturation_mask(
    telemetry: dict[str, Any],
    actions: np.ndarray,
    actuator_force: np.ndarray,
    fraction: float,
) -> np.ndarray:
    saturation = np.zeros((actions.shape[0],), dtype=bool)

    torque_limits = telemetry.get("torque_limits")
    if torque_limits is not None:
        limits = np.asarray(torque_limits, dtype=np.float64)
        valid = np.isfinite(limits) & (limits > 0.0)
        if np.any(valid):
            saturated = np.zeros_like(actuator_force, dtype=bool)
            saturated[:, valid] = np.abs(actuator_force[:, valid]) >= fraction * limits[valid]
            saturation |= np.any(saturated, axis=-1)

    action_limits = telemetry.get("action_limits")
    if action_limits is None:
        return saturation
    limits = np.asarray(action_limits, dtype=np.float64)
    valid = np.isfinite(limits) & (limits > 0.0)
    saturated = np.zeros_like(actions, dtype=bool)
    saturated[:, valid] = np.abs(actions[:, valid]) >= fraction * limits[valid]
    saturation |= np.any(saturated, axis=-1)
    return saturation


def _make_event(
    episode_id: str,
    event_type: str,
    start: int,
    end: int,
    severity: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "event_type": event_type,
        "start_step": int(start),
        "end_step": int(end - 1),
        "duration_steps": int(end - start),
        "severity": float(severity),
        "metrics": metrics,
    }


class GoldenDatasetCurator:
    """Collects, summarizes and persists long-tail evaluation events."""

    def __init__(self, thresholds: CurationThresholds | None = None) -> None:
        self.thresholds = thresholds or CurationThresholds()
        self.events: list[dict[str, Any]] = []
        self.episode_summaries: list[dict[str, Any]] = []

    def add_episode(
        self, telemetry: dict[str, Any], episode_id: str = "episode_0"
    ) -> list[dict[str, Any]]:
        events = categorize_episode(telemetry, self.thresholds, episode_id)
        self.events.extend(events)
        self.episode_summaries.append(self._summarize_episode(telemetry, episode_id, events))
        return events

    def as_records(self) -> list[dict[str, Any]]:
        return list(self.events)

    def report(self) -> dict[str, Any]:
        counts = {event_type: 0 for event_type in EDGE_CASE_TAXONOMY}
        for event in self.events:
            counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1

        total_steps = sum(summary["num_steps"] for summary in self.episode_summaries)
        done_steps = sum(summary["done_steps"] for summary in self.episode_summaries)
        reward_values = [summary["reward_mean"] for summary in self.episode_summaries]
        return {
            "num_episodes": len(self.episode_summaries),
            "num_events": len(self.events),
            "event_counts": counts,
            "stability_percent": 100.0 * (1.0 - done_steps / max(total_steps, 1)),
            "episode_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
            "episode_reward_variance": float(np.var(reward_values)) if reward_values else 0.0,
        }

    def save_dataset(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "taxonomy": EDGE_CASE_TAXONOMY,
            "thresholds": asdict(self.thresholds),
            "events": self.events,
            "episode_summaries": self.episode_summaries,
            "report": self.report(),
        }
        if path.suffix == ".json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        else:
            with path.open("wb") as f:
                pickle.dump(payload, f)
        return path

    @staticmethod
    def _summarize_episode(
        telemetry: dict[str, Any],
        episode_id: str,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rewards = _as_array(telemetry, "rewards", [])
        dones = _as_array(telemetry, "dones", np.zeros((len(rewards),), dtype=bool)).astype(bool)
        return {
            "episode_id": episode_id,
            "num_steps": int(len(rewards)),
            "done_steps": int(np.sum(dones)),
            "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
            "reward_variance": float(np.var(rewards)) if len(rewards) else 0.0,
            "num_events": len(events),
        }
