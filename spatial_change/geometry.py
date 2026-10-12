"""Similarity alignment, cameras, and voxel clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity that maps src (N,3) onto dst (N,3): dst ~= scale * R @ src + t.

    Returns scale, rotation (3,3), and translation (3,).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("umeyama expects matching (N,3) point sets")
    if len(src) < 3:
        raise ValueError("umeyama needs at least 3 points")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    if var_src < 1e-12:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(singular) @ sign) / var_src)
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_sim3(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (scale * (rotation @ points.T)).T + translation


def ransac_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    iterations: int = 800,
    seed: int = 0,
    min_inliers: int = 12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """Robust Sim3. Returns scale, R, t, inlier mask, or None if alignment fails."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    count = len(src)
    if count < min_inliers:
        return None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        choice = rng.choice(count, size=5, replace=False)
        try:
            scale, rotation, translation = umeyama(src[choice], dst[choice])
        except np.linalg.LinAlgError:
            continue
        if not (0.05 <= scale <= 20.0) or not np.isfinite(scale):
            continue
        residual = np.linalg.norm(apply_sim3(src, scale, rotation, translation) - dst, axis=1)
        mask = residual < threshold
        inlier_count = int(mask.sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best_mask = mask
    if best_mask is None or best_count < min_inliers:
        return None
    scale, rotation, translation = umeyama(src[best_mask], dst[best_mask])
    refined = np.linalg.norm(apply_sim3(src, scale, rotation, translation) - dst, axis=1)
    best_mask = refined < threshold
    if int(best_mask.sum()) < min_inliers:
        return None
    scale, rotation, translation = umeyama(src[best_mask], dst[best_mask])
    return scale, rotation, translation, best_mask


def camera_center(w2c: np.ndarray) -> np.ndarray:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return -rotation.T @ translation


def project_points(
    points: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points. Returns pixels (N,2), camera-frame depth, and a valid mask."""
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    camera = (rotation @ np.asarray(points, dtype=np.float64).T).T + translation
    depth = camera[:, 2]
    valid = depth > 1e-6
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    if np.any(valid):
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        pixels[valid, 0] = fx * camera[valid, 0] / depth[valid] + cx
        pixels[valid, 1] = fy * camera[valid, 1] / depth[valid] + cy
    return pixels, depth, valid


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = fx
    matrix[1, 1] = fy
    matrix[0, 2] = cx
    matrix[1, 2] = cy
    return matrix


def estimate_intrinsics(width: int, height: int, fov_deg: float = 62.0) -> np.ndarray:
    """Pinhole intrinsics from a vertical field of view. Phone main cameras sit near 60–70°."""
    fy = (height / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return intrinsics_matrix(fy, fy, width / 2.0, height / 2.0)


def scale_intrinsics(matrix: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    scaled = matrix.copy()
    scaled[0, 0] *= dst_w / src_w
    scaled[0, 2] *= dst_w / src_w
    scaled[1, 1] *= dst_h / src_h
    scaled[1, 2] *= dst_h / src_h
    return scaled


def look_at(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """World-to-camera matrix. The camera looks along +Z, with +Y pointing down in the image."""
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0])
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(world_up, forward)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, forward)
        norm_right = np.linalg.norm(right)
    right = right / norm_right
    down = np.cross(right, forward)
    rotation_c2w = np.stack([right, down, forward], axis=1)
    rotation = rotation_c2w.T
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ eye
    return w2c


def rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector source onto target."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1e-8:
        if cosine > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - source * np.dot(axis, source)
        axis = axis / np.linalg.norm(axis)
        return _rodrigues(axis, np.pi)
    axis = cross / sine
    return _rodrigues(axis, np.arctan2(sine, cosine))


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def average_camera_up(poses: list[np.ndarray]) -> np.ndarray:
    ups = []
    for w2c in poses:
        rotation = w2c[:3, :3]
        ups.append(rotation.T @ np.array([0.0, -1.0, 0.0]))
    up = np.mean(ups, axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0])
    return up / norm


@dataclass
class VoxelClusters:
    labels: np.ndarray
    clusters: list[np.ndarray]


def cluster_points(points: np.ndarray, voxel: float) -> VoxelClusters:
    """26-connected components on a voxel grid."""
    if len(points) == 0:
        return VoxelClusters(labels=np.zeros(0, dtype=np.int32), clusters=[])
    keys = np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int32)
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def find(key: tuple[int, int, int]) -> tuple[int, int, int]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    occupied: dict[tuple[int, int, int], list[int]] = {}
    for index, key_row in enumerate(keys):
        key = (int(key_row[0]), int(key_row[1]), int(key_row[2]))
        occupied.setdefault(key, []).append(index)
        parent[key] = key

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    for key in list(occupied):
        for dx, dy, dz in offsets:
            neighbor = (key[0] + dx, key[1] + dy, key[2] + dz)
            if neighbor in parent:
                union(key, neighbor)

    groups: dict[tuple[int, int, int], list[int]] = {}
    for key, indices in occupied.items():
        groups.setdefault(find(key), []).extend(indices)

    labels = np.full(len(points), -1, dtype=np.int32)
    clusters: list[np.ndarray] = []
    for label, indices in enumerate(groups.values()):
        arr = np.asarray(indices, dtype=np.int32)
        labels[arr] = label
        clusters.append(arr)
    return VoxelClusters(labels=labels, clusters=clusters)


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average points and colors that fall in the same voxel."""
    if len(points) == 0:
        return points.copy(), colors.copy()
    keys = np.floor(points / voxel).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    points = points[order]
    colors = colors[order]
    change = np.any(np.diff(keys, axis=0) != 0, axis=1)
    starts = np.concatenate([[0], np.flatnonzero(change) + 1])
    ends = np.concatenate([starts[1:], [len(points)]])
    out_points = np.empty((len(starts), 3), dtype=np.float64)
    out_colors = np.empty((len(starts), 3), dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        out_points[i] = points[start:end].mean(axis=0)
        out_colors[i] = colors[start:end].mean(axis=0)
    return out_points, out_colors
