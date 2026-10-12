"""Dense points by photometric plane-sweep between reconstructed keyframes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_change.geometry import camera_center, project_points
from spatial_change.sfm import Reconstruction


@dataclass
class DenseCloud:
    points: np.ndarray
    colors: np.ndarray
    camera_ids: np.ndarray


def densify(
    reconstruction: Reconstruction,
    intrinsics: np.ndarray,
    max_pairs: int = 8,
    max_points_per_pair: int = 4000,
    seed: int = 0,
) -> DenseCloud:
    """Build a semi-dense cloud. Falls back to the sparse map if stereo finds little surface."""
    frames = sorted(reconstruction.poses)
    if len(frames) < 2:
        return _from_sparse(reconstruction)
    median_depth = _median_camera_depth(reconstruction)
    candidates: list[tuple[float, int, int]] = []
    for index, frame_a in enumerate(frames):
        center_a = camera_center(reconstruction.poses[frame_a])
        for frame_b in frames[index + 1 : index + 5]:
            baseline = np.linalg.norm(camera_center(reconstruction.poses[frame_b]) - center_a)
            ideal = 0.14 * median_depth
            if baseline < 0.04 * median_depth or baseline > 0.45 * median_depth:
                continue
            candidates.append((abs(baseline - ideal), frame_a, frame_b))
    candidates.sort()
    chosen = candidates[:max_pairs]
    chunks_p: list[np.ndarray] = []
    chunks_c: list[np.ndarray] = []
    chunks_id: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for _, frame_a, frame_b in chosen:
        points, colors = _stereo_pair(
            reconstruction.images[frame_a],
            reconstruction.images[frame_b],
            reconstruction.poses[frame_a],
            reconstruction.poses[frame_b],
            intrinsics,
            rng,
            max_points_per_pair,
        )
        if len(points) == 0:
            continue
        others = [frame for frame in frames if frame != frame_a and frame != frame_b]
        if others:
            frame_c = min(others, key=lambda frame: abs(frame - frame_a))
            points, colors = _agree_with_view(
                points,
                colors,
                reconstruction.images[frame_c],
                reconstruction.poses[frame_c],
                intrinsics,
            )
        if len(points) == 0:
            continue
        chunks_p.append(points)
        chunks_c.append(colors)
        chunks_id.append(np.full(len(points), frame_a, dtype=np.int32))
    if not chunks_p:
        return _from_sparse(reconstruction)
    cloud = DenseCloud(
        points=np.concatenate(chunks_p),
        colors=np.concatenate(chunks_c),
        camera_ids=np.concatenate(chunks_id),
    )
    if len(cloud.points) < 500:
        sparse = _from_sparse(reconstruction)
        return DenseCloud(
            points=np.concatenate([cloud.points, sparse.points]),
            colors=np.concatenate([cloud.colors, sparse.colors]),
            camera_ids=np.concatenate([cloud.camera_ids, sparse.camera_ids]),
        )
    return cloud


def _from_sparse(reconstruction: Reconstruction) -> DenseCloud:
    ids = np.array(
        [frame if frame >= 0 else next(iter(reconstruction.poses)) for frame, _ in reconstruction.observations],
        dtype=np.int32,
    )
    return DenseCloud(
        points=reconstruction.points.copy(),
        colors=np.clip(reconstruction.colors, 0.0, 1.0),
        camera_ids=ids,
    )


def _median_camera_depth(reconstruction: Reconstruction) -> float:
    if len(reconstruction.points) == 0:
        return 2.0
    depths = []
    for w2c in reconstruction.poses.values():
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        z = (rotation @ reconstruction.points.T).T[:, 2] + translation[2]
        z = z[z > 0]
        if len(z):
            depths.append(np.median(z))
    if not depths:
        return 2.0
    return float(np.median(depths))


def _stereo_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    w2c_a: np.ndarray,
    w2c_b: np.ndarray,
    intrinsics: np.ndarray,
    rng: np.random.Generator,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth from photometric plane-sweep between two posed frames.

    Rectified block matching is brittle on handheld clips, so each pixel in the
    first frame is tested at a set of depths and kept when one depth matches the
    second frame clearly better than the others.
    """
    del rng  # the sweep is deterministic; the caller still passes a generator
    height, width = image_a.shape[:2]
    if np.linalg.norm(camera_center(w2c_a) - camera_center(w2c_b)) < 1e-3:
        return np.zeros((0, 3)), np.zeros((0, 3))
    step = 4
    grid_y, grid_x = np.mgrid[6:height - 6:step, 6:width - 6:step]
    xs = grid_x.reshape(-1).astype(np.float64)
    ys = grid_y.reshape(-1).astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ray_x = (xs - cx) / fx
    ray_y = (ys - cy) / fy
    depths = np.linspace(0.55, 3.15, 52)
    rotation = w2c_a[:3, :3]
    translation = w2c_a[:3, 3]
    rotation_t = rotation.T
    offset = rotation_t @ translation
    costs = np.full((len(depths), len(xs)), 1e6, dtype=np.float64)
    image_a_i = image_a.astype(np.int16)
    image_b_i = image_b.astype(np.int16)
    for index, depth in enumerate(depths):
        camera = np.stack([ray_x * depth, ray_y * depth, np.full(len(xs), depth)], axis=1)
        world = (rotation_t @ camera.T).T - offset
        pixels, z, valid = project_points(world, w2c_b, intrinsics)
        ui = np.rint(pixels[:, 0]).astype(np.int32)
        vi = np.rint(pixels[:, 1]).astype(np.int32)
        inside = valid & (z > 0.2) & (ui > 1) & (vi > 1) & (ui < width - 2) & (vi < height - 2)
        costs[index] = _window_cost(image_a_i, image_b_i, ys.astype(np.int32), xs.astype(np.int32), vi, ui, inside)
    best_index = np.argmin(costs, axis=0)
    best_cost = costs[best_index, np.arange(len(xs))]
    masked = costs.copy()
    for delta in (-2, -1, 0, 1, 2):
        neighbor = np.clip(best_index + delta, 0, len(depths) - 1)
        masked[neighbor, np.arange(len(xs))] = 1e9
    second = masked.min(axis=0)
    keep = (best_cost < 30.0) & ((second - best_cost) > 18.0)
    keep &= _depth_agrees(xs, ys, depths[best_index], step)
    if int(keep.sum()) < 50:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if int(keep.sum()) > max_points:
        chosen = np.flatnonzero(keep)
        chosen = chosen[np.linspace(0, len(chosen) - 1, max_points).astype(int)]
        keep = np.zeros_like(keep)
        keep[chosen] = True
    depth = depths[best_index[keep]]
    camera = np.stack([ray_x[keep] * depth, ray_y[keep] * depth, depth], axis=1)
    world = (rotation_t @ camera.T).T - offset
    colors = image_a[ys[keep].astype(int), xs[keep].astype(int)].astype(np.float64) / 255.0
    return world, colors


def _depth_agrees(xs: np.ndarray, ys: np.ndarray, depth: np.ndarray, step: int) -> np.ndarray:
    """Keep pixels whose depth matches the local surface. Isolated wrong depths are dropped."""
    from scipy.ndimage import median_filter

    if len(xs) < 30:
        return np.ones(len(xs), dtype=bool)
    gx = np.rint((xs - xs.min()) / step).astype(np.int32)
    gy = np.rint((ys - ys.min()) / step).astype(np.int32)
    height = int(gy.max()) + 1
    width = int(gx.max()) + 1
    accum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    np.add.at(accum, (gy, gx), depth)
    np.add.at(count, (gy, gx), 1.0)
    valid = count > 0
    image = np.zeros((height, width), dtype=np.float64)
    image[valid] = accum[valid] / count[valid]
    fill = float(np.median(depth))
    smoothed = median_filter(np.where(valid, image, fill), size=5)
    return np.abs(depth - smoothed[gy, gx]) <= 0.12


def _agree_with_view(points, colors, image, w2c, intrinsics, thresh: float = 75.0):
    """Drop points whose colour does not match a third view. A wrong depth rarely agrees twice."""
    if len(points) == 0:
        return points, colors
    height, width = image.shape[:2]
    pixels, depth, valid = project_points(points, w2c, intrinsics)
    ui = np.rint(pixels[:, 0]).astype(np.int32)
    vi = np.rint(pixels[:, 1]).astype(np.int32)
    inside = valid & (depth > 0.2) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
    if int(inside.sum()) < 30:
        return points, colors
    sampled = image[vi[inside], ui[inside]].astype(np.float64)
    error = np.abs(sampled - colors[inside] * 255.0).sum(axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[np.flatnonzero(inside)[error >= thresh]] = False
    if int(keep.sum()) < 30:
        return points, colors
    return points[keep], colors[keep]


def _window_cost(image_a, image_b, ys, xs, vi, ui, inside) -> np.ndarray:
    """Mean absolute RGB difference over a 3x3 window. Lower is a better depth."""
    height, width = image_a.shape[:2]
    total = np.zeros(len(xs), dtype=np.float64)
    count = np.zeros(len(xs), dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ya = ys + dy
            xa = xs + dx
            yb = vi + dy
            xb = ui + dx
            ok = inside & (ya >= 0) & (xa >= 0) & (ya < height) & (xa < width)
            ok &= (yb >= 0) & (xb >= 0) & (yb < height) & (xb < width)
            if not np.any(ok):
                continue
            difference = np.abs(image_a[ya[ok], xa[ok]] - image_b[yb[ok], xb[ok]]).sum(axis=1)
            total[ok] += difference
            count[ok] += 1.0
    cost = np.full(len(xs), 1e6, dtype=np.float64)
    good = count >= 6
    cost[good] = total[good] / count[good]
    return cost
