"""Reference scenes for the independent visual perception layer."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from chimesim.environment import Environment
from chimesim.perception import perceive


def agents(points, size=6, heading=0):
    return [SimpleNamespace(unique_id=i, x=x, y=y, dx=math.cos(heading), dy=math.sin(heading),
                            heading=heading, type=SimpleNamespace(size=size)) for i, (x, y) in enumerate(points)]


class PerceptionTests(unittest.TestCase):
    def test_circle_angular_extent_matches_analytic_value(self):
        result = perceive(agents([(0, 0), (10, 0)]), body_style='Circle with tail', fov_deg=360)
        self.assertAlmostEqual(result.angular_extent[0, 1], 2 * math.asin(.2))
        self.assertAlmostEqual(result.visible_fraction[0, 1], 1)
        self.assertFalse(result.visible.diagonal().any())

    def test_near_circle_fully_occludes_far_circle(self):
        result = perceive(agents([(0, 0), (10, 0), (20, 0)]), body_style='Circle with tail', fov_deg=360)
        self.assertEqual(result.visible_fraction[0, 2], 0)
        self.assertAlmostEqual(result.visible_fraction[0, 1], 1)

    def test_partial_occlusion_and_switch(self):
        scene = agents([(0, 0), (10, 1.8), (20, 0)])
        result = perceive(scene, body_style='Circle with tail', fov_deg=360, samples_per_body=256)
        self.assertGreater(result.visible_fraction[0, 2], 0)
        self.assertLess(result.visible_fraction[0, 2], 1)
        clear = perceive(scene, body_style='Circle with tail', fov_deg=360, agent_occlusion=False)
        self.assertAlmostEqual(clear.visible_fraction[0, 2], 1)

    def test_fov_clips_body_not_just_centre(self):
        result = perceive(agents([(0, 0), (0, 10)]), body_style='Circle with tail', fov_deg=180)
        self.assertAlmostEqual(result.visible_fraction[0, 1], .5)
        zero = perceive(agents([(0, 0), (10, 0)]), fov_deg=0)
        self.assertFalse(zero.visible.any())
        self.assertGreater(zero.angular_extent[0, 1], 0)

    def test_triangle_and_mixed_silhouettes(self):
        result = perceive(agents([(0, 0), (10, 0), (20, 0)]), fov_deg=360)
        self.assertAlmostEqual(result.visible_fraction[0, 1], 1)
        self.assertEqual(result.visible_fraction[0, 2], 0)
        mixed = perceive(agents([(0, 0), (10, 0)]), body_style=['Triangle', 'Circle with tail'], fov_deg=360)
        self.assertAlmostEqual(mixed.angular_extent[0, 1], 2 * math.asin(.2))

    def test_periodic_seam_and_angle_wrap(self):
        scene = agents([(98, 50), (8, 50)])
        result = perceive(scene, Environment(100, 100, bounded=False), body_style='Circle with tail', fov_deg=90)
        self.assertAlmostEqual(result.distance[0, 1], 10)
        self.assertAlmostEqual(result.visible_fraction[0, 1], 1)
        rear = perceive(agents([(20, 20), (10, 20)], heading=math.pi), fov_deg=60)
        self.assertAlmostEqual(rear.visible_fraction[0, 1], 1)

    def test_full_and_partial_wall_occlusion(self):
        scene = agents([(20, 50), (80, 50)], size=30)
        env = Environment(100, 100, wall_shapes=[dict(kind='Rectangle', points=[[.48, 0], [.52, 1]])])
        full = perceive(scene, env, body_style='Circle with tail', fov_deg=360)
        self.assertEqual(full.visible_fraction[0, 1], 0)
        env.set_walls(shapes=[dict(kind='Rectangle', points=[[.48, 0], [.52, .48]])])
        partial = perceive(scene, env, body_style='Circle with tail', fov_deg=360, samples_per_body=256)
        self.assertGreater(partial.visible_fraction[0, 1], .3)
        self.assertLess(partial.visible_fraction[0, 1], .7)

    def test_empty_single_cutoff_and_stationary_heading(self):
        self.assertEqual(perceive([]).visible.shape, (0, 0))
        self.assertFalse(perceive(agents([(0, 0)])).visible.any())
        scene = agents([(0, 0), (10, 0)])
        before = [(a.x, a.y, a.dx, a.dy) for a in scene]
        self.assertFalse(perceive(scene, max_distance=5).visible.any())
        scene[0].dx = scene[0].dy = 0
        self.assertTrue(perceive(scene).visible[0, 1])
        self.assertEqual(before[1], (scene[1].x, scene[1].y, scene[1].dx, scene[1].dy))

    def test_observer_specific_detection_ranges(self):
        result = perceive(agents([(0, 0), (10, 0)]), body_style='Circle with tail',
                          fov_deg=360, max_distance=[0, 20])
        self.assertFalse(result.visible[0, 1])
        self.assertTrue(result.visible[1, 0])

    def test_invalid_controls(self):
        scene = agents([(0, 0), (10, 0)])
        for kwargs in [dict(fov_deg=361), dict(max_distance=-1), dict(samples_per_body=1),
                       dict(body_style='fish'), dict(angular_resolution_deg=0)]:
            with self.assertRaises(ValueError):
                perceive(scene, **kwargs)
