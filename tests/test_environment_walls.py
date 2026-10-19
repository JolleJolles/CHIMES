"""Wall geometry, visibility, movement and grouping regression checks."""
import unittest
from types import SimpleNamespace
import numpy as np
from chimesim.environment import Environment
from chimesim.metrics import spatial_groups


class EnvironmentWallTests(unittest.TestCase):
    def test_sparse_visibility_matches_full_checks_and_traces_only_candidates(self):
        from unittest.mock import patch
        env = self.barrier(False)
        points = np.array([[10., 20.], [90., 20.], [40., 50.], [60., 50.]])
        candidates = np.array([[False, True, False, False], [False, False, True, False]])
        full = env.los_mask(points[:2], points)
        with patch.object(env, "_trace", wraps=env._trace) as trace:
            sparse = env.los_mask(points[:2], points, candidates=candidates)
            self.assertEqual(len(trace.call_args[0][0]), 2)
        np.testing.assert_array_equal(sparse, full & candidates)
        with patch.object(env, "_trace", wraps=env._trace) as trace:
            empty = env.los_mask(points[:2], points, candidates=np.zeros((2, 4), dtype=bool))
            trace.assert_not_called()
            self.assertFalse(empty.any())

    def test_outer_wall_following_preserves_avoidance_and_travel_direction(self):
        points = np.array([[5., 50.], [95., 50.], [50., 5.], [50., 95.], [50., 50.]])
        initial = np.array([[0., 2.], [0., -2.], [2., 0.], [-2., 0.], [1., 1.]])
        base, following = initial.copy(), initial.copy()
        Environment(100, 100, avoid_radius=20, avoid_strength=4).apply_forces(points, base, 100, 100)
        Environment(100, 100, avoid_radius=20, avoid_strength=4,
                    thigmotaxis=2).apply_forces(points, following, 100, 100)
        np.testing.assert_allclose(following - base, [[0, 6], [0, -6], [6, 0], [-6, 0], [0, 0]])
        # Following adds no normal attraction and does nothing away from walls.
        np.testing.assert_allclose(following[[0, 1], 0], base[[0, 1], 0])
        np.testing.assert_allclose(following[[2, 3], 1], base[[2, 3], 1])

    def test_outer_wall_following_is_inactive_without_outer_walls(self):
        velocity = np.array([[0., 2.]])
        Environment(100, 100, bounded=False, thigmotaxis=2).apply_forces(
            np.array([[5., 50.]]), velocity, 100, 100)
        np.testing.assert_array_equal(velocity, [[0, 2]])

    def test_fill_uses_arena_edges_and_spawns_only_in_white(self):
        from chimesim.environment_drawing import fill_region
        env = Environment(100, 100, bounded=False, wall_shapes=[
            dict(kind='Brush', points=[[.5, 0], [.5, 1]], width=.08)])
        original = env._wall.copy()
        fill = fill_region(env, [.25, .5])
        env.set_walls(shapes=env.wall_shapes + [fill])
        self.assertTrue(env._wall[:, :10].all())
        self.assertFalse(env._wall[:, 20:].any())
        points = env.sample_positions(1000, np.random.default_rng(3))
        self.assertTrue((points[:, 0] > 50).all())
        undo_fill = fill_region(env, [.25, .5])
        env.set_walls(shapes=env.wall_shapes + [undo_fill])
        np.testing.assert_array_equal(env._wall, original)
        env.resize(200, 200)
        self.assertTrue(env.has_mask)

    def test_closed_brush_fill_preserves_outline_on_second_click(self):
        from chimesim.environment_drawing import fill_region
        env = Environment(100, 100, wall_shapes=[dict(kind='Brush',
            points=[[.2, .2], [.8, .2], [.8, .8], [.2, .8], [.2, .2]], width=.08)])
        original = env._wall.copy()
        env.set_walls(shapes=env.wall_shapes + [fill_region(env, [.5, .5])])
        self.assertTrue(env._wall[12, 12])
        env.set_walls(shapes=env.wall_shapes + [fill_region(env, [.5, .5])])
        np.testing.assert_array_equal(env._wall, original)

    def test_nearly_closed_circle_fills_without_leaking_and_renders_smoothly(self):
        from chimesim.environment_drawing import fill_region, _filled_image
        angles = np.linspace(0, 2 * np.pi - .08, 100)
        points = np.column_stack((.5 + .35 * np.cos(angles), .5 + .35 * np.sin(angles))).tolist()
        env = Environment(400, 400, wall_shapes=[dict(kind='Brush', points=points, width=.015)])
        original_visual = np.array(_filled_image(env))
        env.set_walls(shapes=env.wall_shapes + [fill_region(env, [.5, .5])])
        self.assertTrue(env._wall[50, 50])
        self.assertFalse(env._wall[0, 0])
        visual = np.array(_filled_image(env))
        self.assertEqual(visual[400, 400], 0)
        self.assertEqual(visual[0, 0], 255)
        env.set_walls(shapes=env.wall_shapes + [fill_region(env, [.5, .5])])
        np.testing.assert_array_equal(np.array(_filled_image(env)), original_visual)

    def test_legacy_fill_reconstructs_smooth_boundary_and_toggles(self):
        from chimesim.environment_drawing import fill_region, _filled_image
        angles = np.linspace(0, 2 * np.pi, 90)
        points = np.column_stack((.5 + .32 * np.cos(angles), .5 + .32 * np.sin(angles))).tolist()
        env = Environment(200, 200, wall_shapes=[dict(kind='Brush', points=points, width=.04)])
        baseline = np.array(_filled_image(env))
        fill = fill_region(env, [.05, .05])
        modern = dict(fill)
        fill.pop('seed')
        outline = list(env.wall_shapes)
        env.set_walls(shapes=outline + [modern])
        expected = np.array(_filled_image(env))
        env.set_walls(shapes=outline + [fill])
        np.testing.assert_array_equal(np.array(_filled_image(env)), expected)
        erased = dict(fill, solid=False)
        env.set_walls(shapes=outline + [fill, erased])
        np.testing.assert_array_equal(np.array(_filled_image(env)), baseline)

    def test_display_contours_preserve_holes_and_touching_components(self):
        from PIL import Image
        from chimesim.environment_drawing import _image_contours
        image = np.full((20, 20), 255, dtype=np.uint8)
        image[2:15, 2:15] = 0
        image[5:9, 5:9] = 255
        image[15:18, 15:18] = 0  # Diagonal contact only.
        contours = _image_contours(Image.fromarray(image))
        self.assertEqual(len(contours), 3)
        self.assertTrue(any(area < 0 for area, _ in contours))
        self.assertEqual(sum(area / 2 for area, _ in contours), int((image == 0).sum()))

    def barrier(self, bounded=True):
        return Environment(100, 100, bounded=bounded, wall_shapes=[
            dict(kind='Rectangle', points=[[.48, 0], [.52, 1]])])

    def test_wall_blocks_sight_movement_and_group_edges(self):
        env = self.barrier()
        self.assertFalse(env.los_mask([[40, 50]], [[60, 50]])[0, 0])
        points = env.move(np.array([[40., 50.]]), np.array([[30., 0.]]))
        self.assertLess(points[0, 0], 48)
        agents = [SimpleNamespace(x=x, y=y) for x, y in [(40, 50), (40, 52), (60, 50), (60, 52)]]
        self.assertEqual(spatial_groups(agents, env, link_distance=25)['groups'], 2)
        self.assertEqual(spatial_groups(agents, Environment(100, 100), link_distance=25)['groups'], 1)

    def test_periodic_links_use_wrapped_sight(self):
        env = self.barrier(False)
        agents = [SimpleNamespace(x=x, y=50) for x in [2, 98]]
        self.assertEqual(spatial_groups(agents, env, link_distance=5)['groups'], 1)
        env.set_walls(shapes=[dict(kind='Rectangle', points=[[0, 0], [.04, 1]])])
        self.assertEqual(spatial_groups(agents, env, link_distance=5)['groups'], 0)

    def test_resize_preserves_drawing_and_failed_edit_preserves_walls(self):
        env = self.barrier()
        env.resize(200, 200)
        self.assertFalse(env.los_mask([[80, 100]], [[120, 100]])[0, 0])
        before = env._wall.copy()
        with self.assertRaises(ValueError):
            env.set_walls(shapes=[dict(kind='Rectangle', points=[[0, 0], [1, 1]])])
        np.testing.assert_array_equal(env._wall, before)
        env.set_walls()
        self.assertFalse(env.has_mask)

    def test_brush_ellipse_and_projection(self):
        for kind in ['Brush', 'Ellipse']:
            env = Environment(100, 100, wall_shapes=[dict(kind=kind, points=[[.3, .3], [.7, .7]], width=.1)])
            self.assertTrue(env.has_mask)
            point = np.array([[50., 50.]])
            env.project_free(point)
            self.assertFalse(env._wall[int(point[0, 1] * env._gh / 100), int(point[0, 0] * env._gw / 100)])
