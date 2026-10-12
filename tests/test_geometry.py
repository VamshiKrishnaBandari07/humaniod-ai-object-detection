import unittest

import numpy as np

from spatial_change.geometry import (
    apply_sim3,
    intrinsics_matrix,
    look_at,
    project_points,
    ransac_sim3,
    umeyama,
)


class GeometryTests(unittest.TestCase):
    def test_umeyama_recovers_a_known_similarity(self):
        rng = np.random.default_rng(0)
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        source = rng.normal(size=(40, 3))
        destination = apply_sim3(source, 2.0, rotation, np.array([1.0, -2.0, 0.4]))
        scale, estimated_r, translation = umeyama(source, destination)
        mapped = apply_sim3(source, scale, estimated_r, translation)
        self.assertAlmostEqual(scale, 2.0, places=5)
        self.assertLess(np.max(np.abs(mapped - destination)), 1e-6)

    def test_ransac_ignores_outliers(self):
        rng = np.random.default_rng(1)
        rotation = np.eye(3)
        source = rng.normal(size=(60, 3))
        destination = apply_sim3(source, 1.4, rotation, np.array([0.2, 0.0, -0.3]))
        source[:8] += rng.normal(scale=3.0, size=(8, 3))
        estimated = ransac_sim3(source, destination, threshold=0.05, iterations=400, seed=1)
        self.assertIsNotNone(estimated)
        scale, estimated_r, translation, inliers = estimated
        self.assertGreater(int(inliers.sum()), 40)
        self.assertAlmostEqual(scale, 1.4, places=2)
        mapped = apply_sim3(source[inliers], scale, estimated_r, translation)
        self.assertLess(np.median(np.linalg.norm(mapped - destination[inliers], axis=1)), 0.02)

    def test_look_at_puts_the_target_on_the_principal_point(self):
        intrinsics = intrinsics_matrix(500, 500, 320, 180)
        pose = look_at(np.array([0.0, 1.5, 2.0]), np.array([0.0, 0.5, 0.0]))
        target = np.array([[0.0, 0.5, 0.0]])
        above = np.array([[0.0, 1.3, 0.0]])
        pixels, depth, valid = project_points(target, pose, intrinsics)
        pixels_above, _, _ = project_points(above, pose, intrinsics)
        self.assertTrue(valid[0])
        self.assertGreater(depth[0], 0.0)
        self.assertLess(abs(pixels[0, 0] - 320), 1.5)
        self.assertLess(abs(pixels[0, 1] - 180), 1.5)
        self.assertLess(pixels_above[0, 1], pixels[0, 1])


if __name__ == "__main__":
    unittest.main()
