import unittest

import numpy as np

from spatial_change.changes import detect_changes


def _box(center, size, color, rng, count=500):
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0
    points = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            face = rng.uniform(-half, half, size=(count // 6, 3))
            face[:, axis] = sign * half[axis]
            points.append(face + center)
    pts = np.concatenate(points)
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(pts), 1))
    return pts, colors


class ChangeTests(unittest.TestCase):
    def test_identical_clouds_have_no_changes(self):
        rng = np.random.default_rng(0)
        points = rng.uniform(-1.0, 1.0, size=(800, 3))
        colors = np.full((800, 3), 0.5)
        result = detect_changes(points, colors, points + 0.001, colors, voxel=0.08, min_cluster_points=20)
        self.assertEqual(result.changes, [])

    def test_move_add_and_remove(self):
        rng = np.random.default_rng(2)
        floor, floor_color = _box((0.0, 0.0, 0.0), (2.4, 0.02, 2.4), (0.6, 0.55, 0.45), rng, count=900)
        wall, wall_color = _box((0.0, 0.6, -1.1), (2.2, 1.2, 0.04), (0.7, 0.7, 0.72), rng, count=600)
        red_a, red_c = _box((-0.55, 0.22, 0.2), (0.28, 0.36, 0.28), (0.82, 0.12, 0.10), rng)
        red_b, _ = _box((0.60, 0.22, -0.15), (0.28, 0.36, 0.28), (0.82, 0.12, 0.10), rng)
        blue, blue_c = _box((-0.2, 0.2, 0.7), (0.30, 0.34, 0.30), (0.15, 0.30, 0.82), rng)
        green, green_c = _box((0.15, 0.2, -0.55), (0.30, 0.34, 0.30), (0.14, 0.64, 0.22), rng)
        static_p = np.concatenate([floor, wall])
        static_c = np.concatenate([floor_color, wall_color])
        points_a = np.concatenate([static_p, red_a, blue])
        colors_a = np.concatenate([static_c, red_c, blue_c])
        points_b = np.concatenate([static_p + rng.normal(scale=0.002, size=static_p.shape), red_b, green])
        colors_b = np.concatenate([static_c, red_c, green_c])

        result = detect_changes(
            points_a, colors_a, points_b, colors_b,
            voxel=0.05, min_cluster_points=30, up=np.array([0.0, 1.0, 0.0]),
        )
        kinds = sorted(change.kind for change in result.changes)
        colors = {change.kind: change.color_name for change in result.changes}
        self.assertEqual(kinds, ["added", "moved", "removed"])
        self.assertEqual(colors["moved"], "red")
        self.assertEqual(colors["removed"], "blue")
        self.assertEqual(colors["added"], "green")
        self.assertIn("moved", result.summary)
        self.assertIn("removed", result.summary)
        self.assertIn("appeared", result.summary)


if __name__ == "__main__":
    unittest.main()
