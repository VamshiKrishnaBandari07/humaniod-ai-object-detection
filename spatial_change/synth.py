"""Synthetic phone-style recordings of a small room, with known changes.

The pictures are the only input the reconstruction sees. Ground truth is written
beside the videos so a run can be scored, and is not read by the mapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spatial_change.geometry import intrinsics_matrix, look_at, project_points


@dataclass
class SceneObject:
    name: str
    center: np.ndarray
    size: np.ndarray
    color: np.ndarray
    seed: int


@dataclass
class SyntheticScene:
    before: list[SceneObject]
    after: list[SceneObject]
    changes: list[dict]


def build_scene() -> SyntheticScene:
    """A short room: one object moves, one leaves, one arrives. The rest stays put."""
    static = [
        SceneObject("floor", np.zeros(3), np.array([3.4, 0.0, 3.4]), np.array([0.62, 0.58, 0.50]), 1),
        SceneObject("wall-left", np.array([-1.7, 1.15, 0.0]), np.array([0.0, 2.3, 3.4]), np.array([0.72, 0.74, 0.78]), 2),
        SceneObject("wall-back", np.array([0.0, 1.15, -1.7]), np.array([3.4, 2.3, 0.0]), np.array([0.78, 0.73, 0.66]), 3),
        SceneObject("wall-right", np.array([1.7, 1.15, 0.0]), np.array([0.0, 2.3, 3.4]), np.array([0.66, 0.70, 0.74]), 4),
        SceneObject("table", np.array([0.0, 0.74, 0.05]), np.array([1.15, 0.06, 0.70]), np.array([0.48, 0.32, 0.18]), 5),
        SceneObject("shelf", np.array([-1.15, 0.45, -0.85]), np.array([0.38, 0.90, 0.32]), np.array([0.42, 0.44, 0.48]), 6),
        SceneObject("bin", np.array([1.05, 0.16, -0.75]), np.array([0.28, 0.32, 0.28]), np.array([0.85, 0.72, 0.18]), 7),
        SceneObject("book", np.array([-0.15, 0.80, -0.10]), np.array([0.18, 0.04, 0.22]), np.array([0.55, 0.28, 0.62]), 8),
    ]
    # Changing objects sit on the table, in view for the whole camera arc.
    red_before = SceneObject("red-box", np.array([-0.38, 0.90, 0.20]), np.array([0.18, 0.18, 0.18]), np.array([0.82, 0.12, 0.10]), 11)
    red_after = SceneObject("red-box", np.array([0.38, 0.90, 0.16]), np.array([0.18, 0.18, 0.18]), np.array([0.82, 0.12, 0.10]), 11)
    blue = SceneObject("blue-box", np.array([0.12, 0.90, -0.16]), np.array([0.18, 0.18, 0.18]), np.array([0.15, 0.30, 0.82]), 12)
    green = SceneObject("green-box", np.array([-0.02, 0.90, 0.22]), np.array([0.18, 0.18, 0.18]), np.array([0.14, 0.64, 0.22]), 13)
    changes = [
        {
            "type": "moved",
            "color": "red",
            "label": "red box",
            "from": red_before.center.tolist(),
            "to": red_after.center.tolist(),
        },
        {
            "type": "removed",
            "color": "blue",
            "label": "blue box",
            "at": blue.center.tolist(),
        },
        {
            "type": "added",
            "color": "green",
            "label": "green box",
            "at": green.center.tolist(),
        },
    ]
    return SyntheticScene(
        before=static + [red_before, blue],
        after=static + [red_after, green],
        changes=changes,
    )


def generate_dataset(
    out_dir: str | Path,
    frames: int = 14,
    width: int = 640,
    height: int = 360,
) -> dict:
    """Write before/after videos, intrinsics, and ground truth. Returns paths and the scene."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    intrinsics = intrinsics_matrix(0.92 * width, 0.92 * width, width / 2.0, height / 2.0)
    before_frames = []
    after_frames = []
    for i in range(frames):
        print(f"  rendering frame {i + 1}/{frames}")
        before_frames.append(render(scene.before, intrinsics, _camera_pose(i, frames, phase=0.0), width, height))
        after_frames.append(render(scene.after, intrinsics, _camera_pose(i, frames, phase=0.18), width, height))
    before_path = _write_video(out / "before.mp4", before_frames)
    after_path = _write_video(out / "after.mp4", after_frames)
    intrinsics_path = out / "intrinsics.json"
    intrinsics_path.write_text(json.dumps({
        "fx": float(intrinsics[0, 0]),
        "fy": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "width": width,
        "height": height,
    }, indent=2), encoding="utf-8")
    gt_path = out / "ground_truth.json"
    gt_path.write_text(json.dumps({
        "units": "meters",
        "changes": scene.changes,
        "note": "World frame of the synthetic room. The reconstruction has its own gauge.",
    }, indent=2), encoding="utf-8")
    # A still from each clip so the report can show the inputs without a video player.
    cv2.imwrite(str(out / "before_frame.png"), cv2.cvtColor(before_frames[len(before_frames) // 2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "after_frame.png"), cv2.cvtColor(after_frames[len(after_frames) // 2], cv2.COLOR_RGB2BGR))
    return {
        "before": str(before_path),
        "after": str(after_path),
        "intrinsics": str(intrinsics_path),
        "ground_truth": str(gt_path),
        "scene": scene,
    }


def render(
    objects: list[SceneObject],
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterise solid, world-locked textured surfaces with a z-buffer."""
    image = np.full((height, width, 3), 28, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    for obj in objects:
        for triangle in _triangles(obj, step=0.34):
            _raster_triangle(image, zbuf, triangle, obj.color, obj.seed, intrinsics, w2c)
    return image


def _camera_pose(index: int, count: int, phase: float) -> np.ndarray:
    """A slow arc at head height, slightly different for each recording."""
    t = index / max(count - 1, 1)
    angle = np.interp(t, [0, 1], [-0.72 + phase, 0.68 + phase])
    radius = 1.72 + 0.08 * np.sin(t * np.pi + phase)
    height = 1.28 + 0.04 * np.cos(t * 5 + phase)
    eye = np.array([np.sin(angle) * radius, height, np.cos(angle) * 1.55 + 0.15])
    target = np.array([0.02 * np.sin(phase), 0.62, 0.02])
    return look_at(eye, target)


def _triangles(obj: SceneObject, step: float) -> list[np.ndarray]:
    triangles = []
    for corners in _faces(obj):
        triangles.extend(_subdivide(corners, step))
    return triangles


def _faces(obj: SceneObject) -> list[list[np.ndarray]]:
    center = np.asarray(obj.center, dtype=np.float64)
    size = np.asarray(obj.size, dtype=np.float64)
    if obj.name == "floor":
        axes = (np.array([size[0], 0.0, 0.0]), np.array([0.0, 0.0, size[2]]))
        return [_corners(center, axes[0], axes[1])]
    if obj.name == "wall-back":
        return [_corners(center, np.array([size[0], 0.0, 0.0]), np.array([0.0, size[1], 0.0]))]
    if obj.name.startswith("wall"):
        return [_corners(center, np.array([0.0, 0.0, size[2]]), np.array([0.0, size[1], 0.0]))]
    hx, hy, hz = size / 2.0
    spans = [
        (np.array([size[0], 0, 0]), np.array([0, size[1], 0]), np.array([0, 0, -hz])),
        (np.array([size[0], 0, 0]), np.array([0, size[1], 0]), np.array([0, 0, hz])),
        (np.array([0, 0, size[2]]), np.array([0, size[1], 0]), np.array([-hx, 0, 0])),
        (np.array([0, 0, size[2]]), np.array([0, size[1], 0]), np.array([hx, 0, 0])),
        (np.array([size[0], 0, 0]), np.array([0, 0, size[2]]), np.array([0, hy, 0])),
        (np.array([size[0], 0, 0]), np.array([0, 0, size[2]]), np.array([0, -hy, 0])),
    ]
    return [_corners(center + offset, axis_u, axis_v) for axis_u, axis_v, offset in spans]


def _corners(center: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> list[np.ndarray]:
    return [
        center - 0.5 * axis_u - 0.5 * axis_v,
        center + 0.5 * axis_u - 0.5 * axis_v,
        center + 0.5 * axis_u + 0.5 * axis_v,
        center - 0.5 * axis_u + 0.5 * axis_v,
    ]


def _subdivide(corners: list[np.ndarray], step: float) -> list[np.ndarray]:
    origin = corners[0]
    axis_u = corners[1] - corners[0]
    axis_v = corners[3] - corners[0]
    nu = max(int(np.linalg.norm(axis_u) / step), 1)
    nv = max(int(np.linalg.norm(axis_v) / step), 1)
    triangles = []
    for i in range(nu):
        for j in range(nv):
            a = origin + axis_u * (i / nu) + axis_v * (j / nv)
            b = origin + axis_u * ((i + 1) / nu) + axis_v * (j / nv)
            c = origin + axis_u * ((i + 1) / nu) + axis_v * ((j + 1) / nv)
            d = origin + axis_u * (i / nu) + axis_v * ((j + 1) / nv)
            triangles.append(np.stack([a, b, c]))
            triangles.append(np.stack([a, c, d]))
    return triangles


def _raster_triangle(image, zbuf, verts, color, seed, intrinsics, w2c) -> None:
    pixels, depth, valid = project_points(verts, w2c, intrinsics)
    if not np.all(valid) or np.any(depth < 0.2):
        return
    height, width = zbuf.shape
    us, vs = pixels[:, 0], pixels[:, 1]
    min_x = max(int(np.floor(us.min())), 0)
    max_x = min(int(np.ceil(us.max())), width - 1)
    min_y = max(int(np.floor(vs.min())), 0)
    max_y = min(int(np.ceil(vs.max())), height - 1)
    if min_x >= max_x or min_y >= max_y:
        return
    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    x0, y0, x1, y1, x2, y2 = us[0], vs[0], us[1], vs[1], us[2], vs[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-6:
        return
    w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denom
    w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not np.any(inside):
        return
    inv_z = w0 / depth[0] + w1 / depth[1] + w2 / depth[2]
    z = np.full(inv_z.shape, np.inf, dtype=np.float32)
    safe = inside & (inv_z > 1e-6)
    z[safe] = (1.0 / inv_z[safe]).astype(np.float32)
    region_z = zbuf[min_y:max_y + 1, min_x:max_x + 1]
    closer = safe & (z < region_z)
    if not np.any(closer):
        return
    b0 = np.zeros_like(w0)
    b1 = np.zeros_like(w1)
    b2 = np.zeros_like(w2)
    b0[safe] = (w0[safe] / depth[0]) / inv_z[safe]
    b1[safe] = (w1[safe] / depth[1]) / inv_z[safe]
    b2[safe] = (w2[safe] / depth[2]) / inv_z[safe]
    world = b0[..., None] * verts[0] + b1[..., None] * verts[1] + b2[..., None] * verts[2]
    shaded = _shade(np.asarray(color, dtype=np.float64), world[closer], seed)
    region = image[min_y:max_y + 1, min_x:max_x + 1]
    region[closer] = np.clip(shaded * 255.0, 0, 255).astype(np.uint8)
    region_z[closer] = z[closer]


def _shade(color: np.ndarray, points: np.ndarray, seed: int) -> np.ndarray:
    """World-locked pattern coarse enough to match across views, fine enough to sit on a small box."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    phase = float(seed) * 0.37
    coarse = np.sin(x * 2.4 + phase) * np.sin(z * 2.1 + phase * 0.5)
    # ~14 cm. Short enough to mark a tabletop object, long enough to survive compression and match.
    mid = np.sin(x * 42.0 + z * 35.0 + y * 28.0 + phase)
    blocks = ((np.floor(x * 3.0 + phase) + np.floor(z * 3.0)) % 2.0) * 2.0 - 1.0
    value = 0.52 + 0.12 * coarse + 0.22 * mid + 0.14 * blocks
    return np.clip(color * value[:, None], 0.0, 1.0)


def _write_video(path: Path, frames: list[np.ndarray]) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    if not writer.isOpened():
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open a video writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path
