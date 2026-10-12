"""Feature-based incremental reconstruction for a short phone clip."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from spatial_change.geometry import camera_center


@dataclass
class Reconstruction:
    images: list[np.ndarray]
    keypoints: list[np.ndarray]
    descriptors: list[np.ndarray | None]
    poses: dict[int, np.ndarray]
    points: np.ndarray
    colors: np.ndarray
    descriptors_3d: np.ndarray
    observations: list[tuple[int, int]] = field(default_factory=list)
    """One (frame, keypoint index) per 3D point, used to match maps together."""


def _sift(max_features: int) -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.03, edgeThreshold=12)


def detect_features(images: list[np.ndarray], max_features: int = 2500) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    extractor = _sift(max_features)
    keypoints: list[np.ndarray] = []
    descriptors: list[np.ndarray | None] = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, desc = extractor.detectAndCompute(gray, None)
        if not found or desc is None:
            keypoints.append(np.zeros((0, 2), dtype=np.float64))
            descriptors.append(None)
            continue
        keypoints.append(np.array([kp.pt for kp in found], dtype=np.float64))
        descriptors.append(desc.astype(np.float32))
    return keypoints, descriptors


def match_descriptors(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    ratio: float = 0.75,
) -> np.ndarray:
    """Return (M,2) int pairs of keypoint indices."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), dtype=np.int32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    kept: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            kept.append((best.queryIdx, best.trainIdx))
    if not kept:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(kept, dtype=np.int32)


def _projection(intrinsics: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return intrinsics @ w2c[:3, :]


def _depths(w2c: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return (rotation @ points.T).T[:, 2] + translation[2]


def _reprojection_error(
    intrinsics: np.ndarray, w2c: np.ndarray, points: np.ndarray, pixels: np.ndarray
) -> np.ndarray:
    camera = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
    depth = camera[:, 2]
    projected = np.zeros_like(pixels)
    valid = depth > 1e-6
    projected[valid, 0] = intrinsics[0, 0] * camera[valid, 0] / depth[valid] + intrinsics[0, 2]
    projected[valid, 1] = intrinsics[1, 1] * camera[valid, 1] / depth[valid] + intrinsics[1, 2]
    error = np.linalg.norm(projected - pixels, axis=1)
    error[~valid] = 1e6
    return error


def triangulate_pair(
    intrinsics: np.ndarray,
    w2c_a: np.ndarray,
    w2c_b: np.ndarray,
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    max_error: float = 3.0,
    min_parallax_deg: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate correspondences. Returns points and a boolean mask over the inputs."""
    if len(pixels_a) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)
    proj_a = _projection(intrinsics, w2c_a)
    proj_b = _projection(intrinsics, w2c_b)
    homog = cv2.triangulatePoints(
        proj_a,
        proj_b,
        pixels_a.T.astype(np.float64),
        pixels_b.T.astype(np.float64),
    )
    points = (homog[:3] / (homog[3] + 1e-12)).T
    depth_a = _depths(w2c_a, points)
    depth_b = _depths(w2c_b, points)
    error_a = _reprojection_error(intrinsics, w2c_a, points, pixels_a)
    error_b = _reprojection_error(intrinsics, w2c_b, points, pixels_b)
    center_a = camera_center(w2c_a)
    center_b = camera_center(w2c_b)
    ray_a = points - center_a
    ray_b = points - center_b
    ray_a /= np.linalg.norm(ray_a, axis=1, keepdims=True) + 1e-12
    ray_b /= np.linalg.norm(ray_b, axis=1, keepdims=True) + 1e-12
    cosine = np.clip(np.sum(ray_a * ray_b, axis=1), -1.0, 1.0)
    parallax = np.degrees(np.arccos(cosine))
    finite = np.isfinite(points).all(axis=1)
    mask = (
        finite
        & (depth_a > 0.05)
        & (depth_b > 0.05)
        & (error_a < max_error)
        & (error_b < max_error)
        & (parallax > min_parallax_deg)
    )
    return points, mask


def _recover_init_pose(
    intrinsics: np.ndarray, pixels_a: np.ndarray, pixels_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(pixels_a) < 20:
        return None
    essential, mask = cv2.findEssentialMat(
        pixels_a,
        pixels_b,
        intrinsics,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.2,
    )
    if essential is None or mask is None or int(mask.sum()) < 20:
        return None
    if essential.shape[0] != 3:
        essential = essential[:3]
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, pixels_a, pixels_b, intrinsics, mask=mask
    )
    if pose_mask is None or int(pose_mask.sum()) < 20:
        return None
    return rotation, translation.reshape(3)


def _sample_color(image: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = int(np.clip(round(float(pixel[0])), 0, width - 1))
    y = int(np.clip(round(float(pixel[1])), 0, height - 1))
    return image[y, x].astype(np.float64) / 255.0


def reconstruct(
    images: list[np.ndarray],
    intrinsics: np.ndarray,
    max_features: int = 2000,
    match_window: int = 4,
    target_depth: float = 2.0,
) -> Reconstruction:
    """Incremental structure from motion. The gauge is fixed so the median depth is `target_depth`."""
    keypoints, descriptors = detect_features(images, max_features=max_features)
    frame_count = len(images)
    pair_matches: dict[tuple[int, int], np.ndarray] = {}
    for i in range(frame_count):
        for j in range(i + 1, min(frame_count, i + match_window + 1)):
            pair_matches[(i, j)] = match_descriptors(descriptors[i], descriptors[j])

    init = _choose_init_pair(pair_matches, keypoints, intrinsics)
    if init is None:
        raise RuntimeError(
            "Could not initialise a map. The clip needs more texture and camera motion."
        )
    frame_a, frame_b, rotation, translation, inlier_pairs = init
    w2c_a = np.eye(4)
    w2c_b = np.eye(4)
    w2c_b[:3, :3] = rotation
    w2c_b[:3, 3] = translation
    points, mask = triangulate_pair(
        intrinsics,
        w2c_a,
        w2c_b,
        keypoints[frame_a][inlier_pairs[:, 0]],
        keypoints[frame_b][inlier_pairs[:, 1]],
    )
    inlier_pairs = inlier_pairs[mask]
    points = points[mask]
    if len(points) < 30:
        raise RuntimeError("Initial triangulation produced too few points.")
    median_depth = float(np.median(_depths(w2c_a, points)))
    scale = target_depth / max(median_depth, 1e-6)
    w2c_b[:3, 3] *= scale
    points = points * scale

    poses: dict[int, np.ndarray] = {frame_a: w2c_a, frame_b: w2c_b}
    point_list = [p for p in points]
    color_list = [_sample_color(images[frame_a], keypoints[frame_a][idx]) for idx in inlier_pairs[:, 0]]
    desc_list = [descriptors[frame_a][idx] for idx in inlier_pairs[:, 0]]
    kp_to_point: dict[tuple[int, int], int] = {}
    for point_id, (ia, ib) in enumerate(inlier_pairs):
        kp_to_point[(frame_a, int(ia))] = point_id
        kp_to_point[(frame_b, int(ib))] = point_id

    remaining = set(range(frame_count)) - set(poses)
    stalled = 0
    while remaining and stalled < len(remaining) + 1:
        ranked: list[tuple[int, np.ndarray, np.ndarray]] = []
        for frame in list(remaining):
            corr = _gather_correspondences(frame, poses, pair_matches, kp_to_point, keypoints, point_list)
            ranked.append((frame, corr[0], corr[1]))
        ranked.sort(key=lambda item: len(item[1]), reverse=True)
        frame, pixels, pts3d = ranked[0]
        if len(pixels) < 15:
            break
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pixels.astype(np.float64),
            intrinsics,
            None,
            iterationsCount=250,
            reprojectionError=4.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 12:
            remaining.remove(frame)
            stalled += 1
            continue
        inlier_ids = inliers.reshape(-1)
        ok, rvec, tvec = cv2.solvePnP(
            pts3d[inlier_ids],
            pixels[inlier_ids],
            intrinsics,
            None,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            remaining.remove(frame)
            stalled += 1
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = tvec.reshape(3)
        poses[frame] = w2c
        remaining.remove(frame)
        stalled = 0
        print(f"  posed frame {frame} ({len(poses)}/{frame_count}, {len(point_list)} points)")
        _triangulate_new_points(
            frame,
            poses,
            pair_matches,
            keypoints,
            descriptors,
            images,
            intrinsics,
            kp_to_point,
            point_list,
            color_list,
            desc_list,
        )

    if len(poses) < 4 or len(point_list) < 80:
        raise RuntimeError(
            f"Reconstruction is too weak ({len(poses)} cameras, {len(point_list)} points). "
            "Use a slower walk around the space with more overlap."
        )
    observations = [(-1, -1)] * len(point_list)
    for (frame, kp_index), point_id in kp_to_point.items():
        if observations[point_id][0] < 0:
            observations[point_id] = (frame, kp_index)
    return Reconstruction(
        images=images,
        keypoints=keypoints,
        descriptors=descriptors,
        poses=poses,
        points=np.asarray(point_list, dtype=np.float64),
        colors=np.asarray(color_list, dtype=np.float64),
        descriptors_3d=np.asarray(desc_list, dtype=np.float32),
        observations=observations,
    )


def _choose_init_pair(
    pair_matches: dict[tuple[int, int], np.ndarray],
    keypoints: list[np.ndarray],
    intrinsics: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pick the frame pair that triangulates the most points, with a mild bias toward a longer baseline."""
    best = None
    best_score = 0.0
    for (i, j), matches in pair_matches.items():
        if len(matches) < 30:
            continue
        pixels_a = keypoints[i][matches[:, 0]]
        pixels_b = keypoints[j][matches[:, 1]]
        recovered = _recover_init_pose(intrinsics, pixels_a, pixels_b)
        if recovered is None:
            continue
        rotation, translation = recovered
        w2c_b = np.eye(4)
        w2c_b[:3, :3] = rotation
        w2c_b[:3, 3] = translation
        _, mask = triangulate_pair(
            intrinsics, np.eye(4), w2c_b, pixels_a, pixels_b, min_parallax_deg=0.6
        )
        inliers = int(mask.sum())
        if inliers < 40:
            continue
        score = inliers + 12.0 * (j - i)
        if score > best_score:
            best_score = score
            best = (i, j, rotation, translation, matches[mask])
    return best


def _gather_correspondences(
    frame: int,
    poses: dict[int, np.ndarray],
    pair_matches: dict[tuple[int, int], np.ndarray],
    kp_to_point: dict[tuple[int, int], int],
    keypoints: list[np.ndarray],
    point_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pixels: list[np.ndarray] = []
    pts: list[np.ndarray] = []
    used: set[int] = set()
    for other in poses:
        key = (min(frame, other), max(frame, other))
        matches = pair_matches.get(key)
        if matches is None:
            continue
        if frame < other:
            src_idx, dst_idx = matches[:, 0], matches[:, 1]
        else:
            src_idx, dst_idx = matches[:, 1], matches[:, 0]
        for kp_frame, kp_other in zip(src_idx, dst_idx):
            point_id = kp_to_point.get((other, int(kp_other)))
            if point_id is None or point_id in used:
                continue
            used.add(point_id)
            pixels.append(keypoints[frame][int(kp_frame)])
            pts.append(point_list[point_id])
    if not pixels:
        return np.zeros((0, 2)), np.zeros((0, 3))
    return np.asarray(pixels, dtype=np.float64), np.asarray(pts, dtype=np.float64)


def _triangulate_new_points(
    frame: int,
    poses: dict[int, np.ndarray],
    pair_matches: dict[tuple[int, int], np.ndarray],
    keypoints: list[np.ndarray],
    descriptors: list[np.ndarray | None],
    images: list[np.ndarray],
    intrinsics: np.ndarray,
    kp_to_point: dict[tuple[int, int], int],
    point_list: list[np.ndarray],
    color_list: list[np.ndarray],
    desc_list: list[np.ndarray],
    max_points: int = 9000,
) -> None:
    if len(point_list) >= max_points:
        return
    for other, pose_other in list(poses.items()):
        if other == frame:
            continue
        key = (min(frame, other), max(frame, other))
        matches = pair_matches.get(key)
        if matches is None or len(matches) == 0:
            continue
        if frame < other:
            idx_frame, idx_other = matches[:, 0], matches[:, 1]
        else:
            idx_frame, idx_other = matches[:, 1], matches[:, 0]
        keep_frame: list[int] = []
        keep_other: list[int] = []
        for a, b in zip(idx_frame, idx_other):
            if (frame, int(a)) in kp_to_point or (other, int(b)) in kp_to_point:
                continue
            keep_frame.append(int(a))
            keep_other.append(int(b))
        if len(keep_frame) < 8:
            continue
        pixels_frame = keypoints[frame][keep_frame]
        pixels_other = keypoints[other][keep_other]
        points, mask = triangulate_pair(
            intrinsics, poses[frame], pose_other, pixels_frame, pixels_other
        )
        for local_i, accept in enumerate(mask):
            if not accept or len(point_list) >= max_points:
                continue
            point_id = len(point_list)
            kp_a = keep_frame[local_i]
            kp_b = keep_other[local_i]
            point_list.append(points[local_i])
            color_list.append(_sample_color(images[frame], keypoints[frame][kp_a]))
            desc = descriptors[frame]
            if desc is None:
                desc_list.append(np.zeros(128, dtype=np.float32))
            else:
                desc_list.append(desc[kp_a])
            kp_to_point[(frame, kp_a)] = point_id
            kp_to_point[(other, kp_b)] = point_id


def match_reconstructions(
    rec_a: Reconstruction,
    rec_b: Reconstruction,
    ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """3D-3D correspondences between two maps, as index pairs into rec_a.points and rec_b.points."""
    if len(rec_a.descriptors_3d) < 8 or len(rec_b.descriptors_3d) < 8:
        return np.zeros((0, 3)), np.zeros((0, 3))
    desc_a, idx_a = _cap_descriptors(rec_a.descriptors_3d, 3500)
    desc_b, idx_b = _cap_descriptors(rec_b.descriptors_3d, 3500)
    pairs = match_descriptors(desc_a, desc_b, ratio=ratio)
    if len(pairs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return rec_a.points[idx_a[pairs[:, 0]]], rec_b.points[idx_b[pairs[:, 1]]]


def _cap_descriptors(descriptors: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if len(descriptors) <= limit:
        return descriptors, np.arange(len(descriptors))
    choice = np.linspace(0, len(descriptors) - 1, limit).astype(int)
    return descriptors[choice], choice
