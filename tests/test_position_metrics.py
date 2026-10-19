"""Analytical checks for within-group position metrics and periodic geometry."""
import unittest
from types import SimpleNamespace

import numpy as np

from chimesim.environment import Environment
from chimesim import metrics


def agents(points, velocity=(1, 0)):
    return [SimpleNamespace(unique_id=i, x=x, y=y, dx=velocity[0], dy=velocity[1])
            for i, (x, y) in enumerate(points)]


class PositionMetricsTests(unittest.TestCase):
    def test_front_back_and_distance_have_known_values(self):
        result = metrics.within_group_positions(agents([(10, 10), (12, 10), (14, 10)]),
                                               Environment(100, 100), link_distance=3)
        np.testing.assert_allclose(result["front_back"], [-2, 0, 2])
        np.testing.assert_allclose(result["front_back_relative"], [-1, 0, 1])
        np.testing.assert_allclose(result["centroid_distance"], [2, 0, 2])
        np.testing.assert_allclose(result["centroid_distance_relative"], [1, 0, 1])
        np.testing.assert_allclose(result["centroid_x"], [12, 12, 12])

    def test_separate_clusters_get_their_own_centroid_and_heading(self):
        first = agents([(10, 10), (12, 10), (14, 10)])
        second = agents([(70, 70), (70, 72), (70, 74)], velocity=(0, 1))
        for i, a in enumerate(second, 3): a.unique_id = i
        result = metrics.within_group_positions(first + second, Environment(100, 100), link_distance=3)
        np.testing.assert_allclose(result["front_back"], [-2, 0, 2, -2, 0, 2])
        self.assertEqual(result["groups"], 2)
        np.testing.assert_array_equal(result["group_size"], np.full(6, 3))

    def test_rotation_and_translation_do_not_change_scores(self):
        reference = metrics.within_group_positions(agents([(10, 10), (12, 10), (14, 10)]),
                                                  Environment(100, 100), link_distance=3)
        rotated = metrics.within_group_positions(agents([(50, 20), (50, 22), (50, 24)], velocity=(0, 2)),
                                                Environment(100, 100), link_distance=3)
        for key in ("front_back", "front_back_relative", "centroid_distance"):
            np.testing.assert_allclose(reference[key], rotated[key])

    def test_periodic_seam_matches_compact_group(self):
        result = metrics.within_group_positions(agents([(98, 10), (0, 10), (2, 10)]),
                                               Environment(100, 100, bounded=False), link_distance=3)
        np.testing.assert_allclose(result["front_back"], [-2, 0, 2], atol=1e-12)
        np.testing.assert_allclose(result["centroid_distance"], [2, 0, 2], atol=1e-12)
        np.testing.assert_allclose(result["centroid_x"] % 100, [0, 0, 0], atol=1e-12)

    def test_four_corners_have_one_unwrapped_centre(self):
        result = metrics.within_group_positions(agents([(1, 1), (99, 1), (1, 99), (99, 99)]),
                                               Environment(100, 100, bounded=False), link_distance=3)
        self.assertEqual(result["groups"], 1)
        np.testing.assert_allclose(result["front_back"], [1, -1, 1, -1], atol=1e-12)
        np.testing.assert_allclose(result["centroid_distance"], np.sqrt(2), atol=1e-12)

    def test_stopped_group_has_distance_but_no_front(self):
        result = metrics.within_group_positions(agents([(10, 10), (12, 10)], velocity=(0, 0)),
                                               Environment(100, 100), link_distance=3)
        self.assertTrue(np.isnan(result["front_back"]).all())
        np.testing.assert_allclose(result["centroid_distance"], [1, 1])

    def test_opposed_headings_do_not_create_a_false_leader(self):
        group = agents([(10, 10), (12, 10)])
        group[1].dx = -1
        result = metrics.within_group_positions(group, Environment(100, 100), link_distance=3)
        self.assertTrue(np.isnan(result["front_back"]).all())
        np.testing.assert_allclose(result["polarisation"], [0, 0])

    def test_singletons_and_empty_groups_are_explicitly_undefined(self):
        result = metrics.within_group_positions(agents([(10, 10)]), Environment(100, 100))
        self.assertEqual(result["component"][0], -1)
        self.assertTrue(np.isnan(result["centroid_distance"][0]))
        self.assertEqual(metrics.within_group_positions([], Environment(100, 100))["groups"], 0)

    def test_ambiguous_periodic_centre_is_not_reported(self):
        result = metrics.within_group_positions(agents([(0, 10), (50, 10)]),
                                               Environment(100, 100, bounded=False), link_distance=51)
        self.assertTrue(np.isnan(result["front_back"]).all())
        self.assertTrue(np.isnan(result["centroid_distance"]).all())

    def test_identity_and_permutation(self):
        group = agents([(10, 10), (12, 10), (14, 10)])
        result = metrics.within_group_positions(group[::-1], Environment(100, 100), link_distance=3)
        np.testing.assert_array_equal(result["id"], [2, 1, 0])
        np.testing.assert_allclose(result["front_back"], [2, 0, -2])

    def test_nearest_neighbour_tree_matches_pairwise_distances(self):
        points = np.random.default_rng(9).uniform(0, 100, (30, 2))
        group = agents(points)
        for bounded in (True, False):
            delta = points[:, None] - points[None, :]
            if not bounded: delta -= np.round(delta / 100) * 100
            distance = np.linalg.norm(delta, axis=2)
            np.fill_diagonal(distance, np.inf)
            self.assertAlmostEqual(metrics.mean_nnd(group, Environment(100, 100, bounded=bounded)),
                                   distance.min(axis=1).mean())

    def test_mean_speed_is_centralised(self):
        group = agents([(10, 10), (12, 10)], velocity=(3, 4))
        self.assertEqual(metrics.mean_speed(group), 5)
        self.assertEqual(metrics.compute({"A1": group}, Environment(100, 100))["A1_speed"], 5)
        self.assertEqual(metrics.mean_speed([]), 0)


if __name__ == "__main__":
    unittest.main()
