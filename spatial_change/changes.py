"""Describe what moved, appeared, or vanished between two aligned clouds."""

from __future__ import annotations

from dataclasses import dataclass, field

import colorsys

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Change:
    kind: str
    color_name: str
    label: str
    description: str
    centroid: np.ndarray
    centroid_to: np.ndarray | None
    points: np.ndarray
    points_to: np.ndarray | None
    size: float
    rgb: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))


@dataclass
class ChangeResult:
    changes: list[Change] = field(default_factory=list)
    summary: str = ""
    tau: float = 0.0
    static_bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    static_bounds_max: np.ndarray = field(default_factory=lambda: np.ones(3))


def color_name(rgb: np.ndarray) -> str:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    red, green, blue = [float(np.clip(channel, 0.0, 1.0)) for channel in rgb[:3]]
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    if value < 0.12:
        return "black"
    if saturation < 0.18:
        return "white" if value > 0.78 else "gray"
    if hue < 0.04 or hue >= 0.96:
        return "red"
    if hue < 0.09:
        return "orange"
    if hue < 0.18:
        return "yellow"
    if hue < 0.25:
        return "brown" if value < 0.55 else "yellow"
    if hue < 0.46:
        return "green"
    if hue < 0.56:
        return "cyan"
    if hue < 0.78:
        return "blue"
    if hue < 0.90:
        return "purple"
    return "red"


def detect_changes(
    points_a: np.ndarray,
    colors_a: np.ndarray,
    points_b: np.ndarray,
    colors_b: np.ndarray,
    voxel: float | None = None,
    min_cluster_points: int = 25,
    up: np.ndarray | None = None,
) -> ChangeResult:
    """Find added, removed, and moved regions between two aligned clouds.

    Saturated points are grouped by colour. A colour in both clouds is a move
    when its centre shifts more than reconstruction drift. A colour that is
    mostly in one cloud was added or removed. Gray, black, and white are ignored.
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    colors_a = _unit_colors(colors_a)
    colors_b = _unit_colors(colors_b)
    if len(points_a) < 20 or len(points_b) < 20:
        result = ChangeResult(summary="Not enough 3D structure was reconstructed to compare the recordings.")
        return result

    if voxel is None or voxel <= 0:
        span = np.maximum(points_a.max(0) - points_a.min(0), points_b.max(0) - points_b.min(0))
        voxel = float(np.linalg.norm(span) / 70.0)
        voxel = float(np.clip(voxel, 0.02, 0.12))

    dist_ab = _nn_distances(points_a, points_b)
    dist_ba = _nn_distances(points_b, points_a)
    noise = float(np.quantile(np.concatenate([dist_ab, dist_ba]), 0.5))
    tau = float(np.clip(max(noise * 3.5, voxel * 2.0), 0.08, 0.28))
    bounds_min, bounds_max = _static_bounds(points_a, dist_ab, tau, points_b, dist_ba)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    up = up / (np.linalg.norm(up) + 1e-12)
    changes = _color_mass_changes(
        points_a, colors_a, points_b, colors_b,
        min_points=min_cluster_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        up=up,
    )

    changes.sort(key=lambda change: (-change.size, change.kind))
    summary = _summarise(changes)
    return ChangeResult(
        changes=changes,
        summary=summary,
        tau=tau,
        static_bounds_min=bounds_min,
        static_bounds_max=bounds_max,
    )


def _color_mass_changes(
    points_a: np.ndarray,
    colors_a: np.ndarray,
    points_b: np.ndarray,
    colors_b: np.ndarray,
    min_points: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    up: np.ndarray,
) -> list[Change]:
    """Group saturated points by colour and see whether that colour moved, appeared, or vanished.

    A colour seen in both recordings is a move only when its centre shifts more
    than the usual reconstruction drift. Floor and wall colours stay put, so
    they drop out. One object per colour: two red boxes would be reported together.
    """
    labels_a = _saturated_labels(colors_a)
    labels_b = _saturated_labels(colors_b)
    changes: list[Change] = []
    for name in sorted(set(labels_a) | set(labels_b)):
        if name in ("gray", "black", "white"):
            continue
        mask_a = labels_a == name
        mask_b = labels_b == name
        pts_a = points_a[mask_a]
        pts_b = points_b[mask_b]
        cols_a = colors_a[mask_a]
        cols_b = colors_b[mask_b]
        enough_a = len(pts_a) >= min_points
        enough_b = len(pts_b) >= min_points
        # A curtain, tile run, or wall is bigger than one object. Leave it unmarked.
        if (len(pts_a) and _extent(pts_a) > 0.9) or (len(pts_b) and _extent(pts_b) > 0.9):
            continue
        if enough_a and enough_b:
            centroid_a = pts_a.mean(axis=0)
            centroid_b = pts_b.mean(axis=0)
            shift = float(np.linalg.norm(centroid_b - centroid_a))
            drift = max(0.34, 0.35 * max(_rms(pts_a), _rms(pts_b)))
            if shift < drift:
                continue
            label = f"{name} object"
            description = (
                f"A {label} was moved from {_where(centroid_a, bounds_min, bounds_max, up)} "
                f"to {_where(centroid_b, bounds_min, bounds_max, up)}, "
                f"a shift of about {shift:.2f} m."
            )
            changes.append(Change(
                kind="moved",
                color_name=name,
                label=label,
                description=description,
                centroid=centroid_a,
                centroid_to=centroid_b,
                points=pts_a,
                points_to=pts_b,
                size=float(max(_extent(pts_a), _extent(pts_b), shift)),
                rgb=_mean_rgb(np.concatenate([cols_a, cols_b])),
            ))
        elif enough_a and len(pts_b) * 4 < len(pts_a):
            changes.append(_one_sided(name, "removed", pts_a, cols_a, bounds_min, bounds_max, up))
        elif enough_b and len(pts_a) * 4 < len(pts_b):
            changes.append(_one_sided(name, "added", pts_b, cols_b, bounds_min, bounds_max, up))
    return changes


def _one_sided(name, kind, points, colors, bounds_min, bounds_max, up) -> Change:
    centroid = points.mean(axis=0)
    label = f"{name} object"
    place = _where(centroid, bounds_min, bounds_max, up)
    if kind == "removed":
        description = f"A {label} {place} was removed."
    else:
        description = f"A {label} appeared {place}."
    return Change(
        kind=kind,
        color_name=name,
        label=label,
        description=description,
        centroid=centroid,
        centroid_to=None,
        points=points,
        points_to=None,
        rgb=_mean_rgb(colors),
        size=_extent(points),
    )


def _saturated_labels(colors: np.ndarray) -> np.ndarray:
    peak = colors.max(axis=1)
    floor = colors.min(axis=1)
    saturation = (peak - floor) / np.maximum(peak, 1e-6)
    labels = np.array(["gray"] * len(colors), dtype=object)
    for index in np.flatnonzero(saturation >= 0.38):
        labels[index] = color_name(colors[index])
    return labels


def _mean_rgb(colors: np.ndarray) -> np.ndarray:
    if len(colors) == 0:
        return np.array([0.8, 0.8, 0.8])
    return np.clip(np.asarray(colors, dtype=np.float64).mean(axis=0), 0.0, 1.0)


def _rms(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _extent(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _unit_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    if colors.size == 0:
        return np.zeros((0, 3))
    if colors.max() > 1.5:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    distances, _ = cKDTree(dst).query(src, k=1)
    return np.asarray(distances, dtype=np.float64)


def _static_bounds(
    points_a: np.ndarray,
    dist_ab: np.ndarray,
    tau: float,
    points_b: np.ndarray,
    dist_ba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    static = points_a[dist_ab <= tau]
    if len(static) < 30:
        static = np.concatenate([points_a, points_b])
    else:
        static_b = points_b[dist_ba <= tau]
        if len(static_b):
            static = np.concatenate([static, static_b])
    return np.quantile(static, 0.02, axis=0), np.quantile(static, 0.98, axis=0)


def _where(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, up: np.ndarray) -> str:
    span = np.maximum(hi - lo, 1e-6)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, up))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, helper)
    right = right / (np.linalg.norm(right) + 1e-12)
    forward = np.cross(right, up)
    local = point - (lo + hi) / 2.0
    rx = float(np.dot(local, right) / (np.dot(span, np.abs(right)) + 1e-6))
    depth_axis = float(np.dot(local, forward) / (np.dot(span, np.abs(forward)) + 1e-6))
    height_axis = float(np.dot(local, up) / (np.dot(span, np.abs(up)) + 1e-6))
    horizontal = "on the left" if rx < -0.18 else "on the right" if rx > 0.18 else "near the center"
    depth = "toward the front" if depth_axis > 0.18 else "toward the back" if depth_axis < -0.18 else "at mid-depth"
    if height_axis < -0.2:
        height = "near the floor"
    elif height_axis > 0.25:
        height = "in the upper part of the scene"
    else:
        height = "around mid-height"
    return f"{horizontal}, {depth}, {height}"


def resync_summary(result: ChangeResult) -> None:
    result.summary = _summarise(result.changes)


def _summarise(changes: list[Change]) -> str:
    if not changes:
        return (
            "The two recordings show the same small space. "
            "No object-sized region was added, removed, or moved."
        )
    counts = {"moved": 0, "added": 0, "removed": 0}
    for change in changes:
        counts[change.kind] += 1
    lead_bits = []
    if counts["moved"]:
        lead_bits.append(f"{counts['moved']} object{'s' if counts['moved'] != 1 else ''} moved")
    if counts["added"]:
        lead_bits.append(f"{counts['added']} added")
    if counts["removed"]:
        lead_bits.append(f"{counts['removed']} removed")
    return "Between the two recordings, " + ", ".join(lead_bits) + ". " + " ".join(change.description for change in changes)

