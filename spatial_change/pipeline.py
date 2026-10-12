"""End-to-end comparison of two recordings."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from spatial_change.changes import ChangeResult, detect_changes
from spatial_change.dense import densify
from spatial_change.geometry import (
    apply_sim3,
    average_camera_up,
    estimate_intrinsics,
    ransac_sim3,
    rotation_aligning_vectors,
    scale_intrinsics,
    voxel_downsample,
)
from spatial_change.sfm import match_reconstructions, reconstruct
from spatial_change.viz import draw_identification, identify_on_frames, write_ply, write_preview, write_report, write_viewer


def run_comparison(
    before: str,
    after: str,
    out_dir: str,
    intrinsics_path: str | None = None,
    fov_deg: float = 62.0,
    max_frames: int = 12,
    ground_truth: str | None = None,
) -> ChangeResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Reading recordings...")
    frames_a = load_video(before, max_frames=max_frames)
    frames_b = load_video(after, max_frames=max_frames)
    intrinsics_a = _intrinsics_for(frames_a, intrinsics_path, fov_deg)
    intrinsics_b = _intrinsics_for(frames_b, intrinsics_path, fov_deg)

    print(f"Reconstructing the first recording ({len(frames_a)} frames)...")
    rec_a = reconstruct(frames_a, intrinsics_a)
    print(f"  {len(rec_a.poses)} cameras, {len(rec_a.points)} sparse points")
    print(f"Reconstructing the second recording ({len(frames_b)} frames)...")
    rec_b = reconstruct(frames_b, intrinsics_b)
    print(f"  {len(rec_b.poses)} cameras, {len(rec_b.points)} sparse points")

    print("Densifying both maps...")
    dense_a = densify(rec_a, intrinsics_a)
    dense_b = densify(rec_b, intrinsics_b)
    dense_a = _keep_near_sparse(dense_a, rec_a.points)
    dense_b = _keep_near_sparse(dense_b, rec_b.points)
    print(f"  dense points: {len(dense_a.points)} and {len(dense_b.points)}")

    print("Aligning the two maps into one frame...")
    points_a, points_b = match_reconstructions(rec_a, rec_b)
    if len(points_a) < 12:
        points_a, points_b = match_reconstructions(rec_a, rec_b, ratio=0.9)
    diagonal = float(np.linalg.norm(rec_a.points.max(0) - rec_a.points.min(0)))
    aligned = ransac_sim3(points_b, points_a, threshold=max(0.08, 0.045 * diagonal), iterations=1200)
    if aligned is None:
        raise RuntimeError(
            "The two maps could not be registered. Record both clips of the same area, "
            "with overlap on the parts of the room that did not move."
        )
    scale, rotation, translation, inliers = aligned
    print(f"  similarity scale {scale:.3f}, {int(inliers.sum())} inliers")

    cloud_b = apply_sim3(dense_b.points, scale, rotation, translation)
    colors_b = dense_b.colors
    cloud_a = dense_a.points
    colors_a = dense_a.colors

    up = average_camera_up(list(rec_a.poses.values()))
    upright = rotation_aligning_vectors(up, np.array([0.0, 0.0, 1.0]))
    cloud_a = cloud_a @ upright.T
    cloud_b = cloud_b @ upright.T
    colors_a, cloud_a = _clean(colors_a, cloud_a)
    colors_b, cloud_b = _clean(colors_b, cloud_b)

    # diagonal is in the first map's gauge. The second cloud has already been scaled into it.
    voxel = float(np.clip(diagonal / 75.0, 0.03, 0.08))
    print("Comparing the aligned clouds...")
    result = detect_changes(
        cloud_a, colors_a, cloud_b, colors_b,
        voxel=voxel, up=np.array([0.0, 0.0, 1.0]),
    )
    before_bgr = cv2.cvtColor(frames_a[len(frames_a) // 2], cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(frames_b[len(frames_b) // 2], cv2.COLOR_RGB2BGR)
    inlier_count = int(inliers.sum())
    marked = None
    if ground_truth is None:
        print("Looking for objects with Grounding DINO, SAM 2, DINOv2, and SigLIP...")
        try:
            from spatial_change.semantic import compare_visits
            semantic = compare_visits(frames_a, frames_b)
        except Exception as exc:
            print(f"Open-set detection did not finish: {exc}")
            semantic = None
        if semantic is not None:
            result, marked_before, marked_after = semantic
            marked = (marked_before, marked_after)
            metadata_note = "Objects come from Grounding DINO boxes, SAM 2 masks, DINOv2 matching, and SigLIP labels."
        else:
            metadata_note = None
    else:
        metadata_note = None
    if marked is None and inlier_count < 80:
        result.changes.clear()
        result.summary = (
            "No object was marked. These two clips do not lock into one map "
            f"({inlier_count} shared places). They take in different parts of the room, "
            "so a box on one picture would not sit on the same thing in the other."
        )
    elif marked is None:
        identify_on_frames(result, before_bgr, after_bgr)
    print(result.summary)

    metadata = {
        "frames_before": len(frames_a),
        "frames_after": len(frames_b),
        "cameras_before": len(rec_a.poses),
        "cameras_after": len(rec_b.poses),
        "sparse_points_before": int(len(rec_a.points)),
        "sparse_points_after": int(len(rec_b.points)),
        "dense_points_before": int(len(cloud_a)),
        "dense_points_after": int(len(cloud_b)),
        "alignment_inliers": int(inliers.sum()),
        "alignment_scale": round(float(scale), 4),
        "change_threshold_m": round(float(result.tau), 4),
        "gauge": "Median sparse depth in each clip is normalised to 2 m, then a similarity aligns the second map onto the first. Units are approximate metres.",
    }
    if metadata_note:
        metadata["objects"] = metadata_note
    if ground_truth:
        metadata["score"] = _score(result, json.loads(Path(ground_truth).read_text(encoding="utf-8")))
        print(
            f"Score against ground truth: {metadata['score']['matched']}"
            f"/{metadata['score']['expected']} changes matched by type and colour."
        )

    _write_outputs(
        out, frames_a, frames_b, cloud_a, colors_a, cloud_b, colors_b, result, metadata, marked=marked,
    )
    print(f"Wrote the report and viewer to {out}")
    return result


def load_video(path: str, max_frames: int = 12, max_side: int = 960) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if total > 0:
        indexes = np.unique(np.linspace(0, total - 1, min(max_frames, total)).astype(int))
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(_resize_rgb(frame, max_side))
    if len(frames) < 4:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        raw: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            raw.append(frame)
        if not raw:
            capture.release()
            raise RuntimeError(f"No frames decoded from {path}")
        pick = np.unique(np.linspace(0, len(raw) - 1, min(max_frames, len(raw))).astype(int))
        frames = [_resize_rgb(raw[int(i)], max_side) for i in pick]
    capture.release()
    if len(frames) < 4:
        raise RuntimeError(f"Need at least 4 frames from {path}")
    return frames


def _resize_rgb(frame: np.ndarray, max_side: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def _intrinsics_for(frames: list[np.ndarray], path: str | None, fov_deg: float) -> np.ndarray:
    height, width = frames[0].shape[:2]
    if path is None:
        return estimate_intrinsics(width, height, fov_deg)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = estimate_intrinsics(1, 1)
    matrix[0, 0] = float(data["fx"])
    matrix[1, 1] = float(data["fy"])
    matrix[0, 2] = float(data["cx"])
    matrix[1, 2] = float(data["cy"])
    src = (int(data.get("width", width)), int(data.get("height", height)))
    if src != (width, height):
        matrix = scale_intrinsics(matrix, src, (width, height))
    return matrix


def _keep_near_sparse(cloud, sparse_points: np.ndarray):
    center = sparse_points.mean(axis=0)
    radius = float(np.linalg.norm(sparse_points - center, axis=1).max())
    distance = np.linalg.norm(cloud.points - center, axis=1)
    mask = distance < radius * 1.8 + 0.5
    if int(mask.sum()) < 200:
        return cloud
    from spatial_change.dense import DenseCloud
    return DenseCloud(points=cloud.points[mask], colors=cloud.colors[mask], camera_ids=cloud.camera_ids[mask])


def _clean(colors: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = _inlier_mask(points)
    points = points[mask]
    colors = colors[mask]
    if len(points) > 35000:
        choice = np.linspace(0, len(points) - 1, 35000).astype(int)
        points = points[choice]
        colors = colors[choice]
    voxel = float(np.clip(np.linalg.norm(points.max(0) - points.min(0)) / 120.0, 0.015, 0.04))
    points, colors = voxel_downsample(points, colors, voxel)
    return colors, points


def _inlier_mask(points: np.ndarray, k: int = 12, std_ratio: float = 3.0) -> np.ndarray:
    if len(points) < k + 5:
        return np.ones(len(points), dtype=bool)
    distances, _ = cKDTree(points).query(points, k=k)
    mean = distances[:, 1:].mean(axis=1)
    limit = float(mean.mean() + std_ratio * mean.std())
    return mean < limit


def _score(result: ChangeResult, ground_truth: dict) -> dict:
    remaining = list(result.changes)
    details = []
    matched = 0
    for expected in ground_truth.get("changes", []):
        hit = None
        for change in remaining:
            if change.kind == expected["type"] and change.color_name == expected["color"]:
                hit = change
                break
        details.append({
            "type": expected["type"],
            "color": expected["color"],
            "matched": hit is not None,
            "description": None if hit is None else hit.description,
        })
        if hit is not None:
            matched += 1
            remaining.remove(hit)
    expected_count = len(ground_truth.get("changes", []))
    return {
        "matched": matched,
        "expected": expected_count,
        "recall": 0.0 if expected_count == 0 else matched / expected_count,
        "extra": [
            {"type": change.kind, "color": change.color_name, "description": change.description}
            for change in remaining
        ],
        "details": details,
    }


def _write_outputs(out, frames_a, frames_b, cloud_a, colors_a, cloud_b, colors_b, result, metadata, marked=None) -> None:
    write_report(out / "report.json", result, metadata)
    write_ply(out / "scene_before.ply", cloud_a, colors_a)
    write_ply(out / "scene_after.ply", cloud_b, colors_b)
    change_points = []
    change_colors = []
    removed_p, removed_c = _coloured([c for c in result.changes if c.kind == "removed"])
    added_p, added_c = _coloured([c for c in result.changes if c.kind == "added"])
    moved_p, moved_c = _coloured([c for c in result.changes if c.kind == "moved"], include_destination=True)
    for pts, cols in ((removed_p, removed_c), (added_p, added_c), (moved_p, moved_c)):
        if len(pts):
            change_points.append(pts)
            change_colors.append(cols)
    if change_points:
        write_ply(out / "changes.ply", np.concatenate(change_points), np.concatenate(change_colors))
    if marked is None:
        before_bgr = cv2.cvtColor(frames_a[len(frames_a) // 2], cv2.COLOR_RGB2BGR)
        after_bgr = cv2.cvtColor(frames_b[len(frames_b) // 2], cv2.COLOR_RGB2BGR)
        marked_before = draw_identification(before_bgr, result, "before")
        marked_after = draw_identification(after_bgr, result, "after")
    else:
        marked_before, marked_after = marked
    cv2.imwrite(str(out / "before_frame.png"), marked_before)
    cv2.imwrite(str(out / "after_frame.png"), marked_after)
    write_preview(out / "preview.png", marked_before, marked_after, cloud_a, colors_a, cloud_b, colors_b, result)
    write_viewer(
        out / "viewer.html",
        {
            "before": (cloud_a, colors_a),
            "after": (cloud_b, colors_b),
            "removed": (removed_p, removed_c),
            "added": (added_p, added_c),
            "moved": (moved_p, moved_c),
        },
        result,
    )


def _coloured(changes, include_destination: bool = False) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    for change in changes:
        colour = np.asarray(change.rgb, dtype=np.float64).reshape(1, 3)
        chunks = [change.points]
        if include_destination and change.points_to is not None:
            chunks.append(change.points_to)
        for chunk in chunks:
            if chunk is None or len(chunk) == 0:
                continue
            points.append(chunk)
            colors.append(np.repeat(colour, len(chunk), axis=0))
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(points), np.concatenate(colors)
