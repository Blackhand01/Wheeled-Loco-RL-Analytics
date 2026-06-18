"""Render a deterministic wheeled-Go1 MuJoCo preview as GIF or MP4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np


def _load_home_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("MJCF model is missing keyframe 'home'.")

    data.qpos[:] = model.key_qpos[home_id]
    data.qvel[:] = model.key_qvel[home_id]
    data.ctrl[:] = model.key_ctrl[home_id]
    mujoco.mj_forward(model, data)
    return model.key_qpos[home_id].copy(), model.key_ctrl[home_id].copy()


def _animate_preview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    home_qpos: np.ndarray,
    home_ctrl: np.ndarray,
    time_s: float,
) -> None:
    qpos = home_qpos.copy()

    qpos[0] = 0.16 * time_s
    qpos[1] = 0.025 * np.sin(2.0 * np.pi * 0.35 * time_s)
    qpos[2] = home_qpos[2] + 0.012 * np.sin(2.0 * np.pi * 1.4 * time_s)

    steering = 0.08 * np.sin(2.0 * np.pi * 0.6 * time_s)
    qpos[[7, 11, 15, 19]] += np.array([steering, -steering, -steering, steering])

    wheel_angle = -12.0 * time_s
    qpos[[10, 14, 18, 22]] = wheel_angle

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = home_ctrl
    mujoco.mj_forward(model, data)


def render_demo(
    model_path: str | Path = ROOT / "envs/assets/scene.xml",
    output: str | Path = ROOT / "media/wheeled_go1_demo.gif",
    width: int = 960,
    height: int = 540,
    fps: int = 24,
    duration: float = 4.0,
) -> Path:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    home_qpos, home_ctrl = _load_home_state(model, data)

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id < 0:
        raise ValueError("MJCF model is missing body 'trunk'.")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.35
    camera.azimuth = 145.0
    camera.elevation = -18.0

    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    frame_count = int(duration * fps)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for frame_idx in range(frame_count):
            time_s = frame_idx / fps
            _animate_preview(model, data, home_qpos, home_ctrl, time_s)

            camera.lookat[:] = data.xpos[trunk_id]
            camera.lookat[2] += 0.02
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    suffix = output.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output, frames, fps=fps, loop=0)
    elif suffix in {".mp4", ".m4v"}:
        imageio.mimsave(output, frames, fps=fps, quality=8, macro_block_size=8)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        imageio.imwrite(output, frames[min(len(frames) // 2, len(frames) - 1)])
    else:
        raise ValueError("Output must end with .gif, .mp4, .png, .jpg or .jpeg.")

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "envs/assets/scene.xml"))
    parser.add_argument("--output", default=str(ROOT / "media/wheeled_go1_demo.gif"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()

    output = render_demo(
        model_path=args.model,
        output=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
