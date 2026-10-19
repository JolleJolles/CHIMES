"""Focused checks for variable-speed responses and periodic spatial grouping.

Run with: python -m unittest discover -s tests
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from chimesim.agents import AgentType
from chimesim.environment import Environment
from chimesim.metrics import spatial_groups
from chimesim.model import Model


class VariableSpeedTests(unittest.TestCase):
    def pair(self, *, separation=80, social_speed=.5, interaction_model="zonal", **parameters):
        kind = AgentType("A1", n_default=2, social_speed=social_speed, noise=0,
                         max_turn=0, r_a=150, interaction_model=interaction_model, **parameters)
        model = Model([kind], seed=9)
        model.env = Environment(1000, 1000, bounded=True, avoid_strength=0)
        model.init_agents("A1", 2)
        for agent, x in zip(model.all_agents, (400, 400 + separation)):
            agent.x, agent.y, agent.dx, agent.dy = x, 500., 3., 0.
        return model

    def advance(self, model, fov=360):
        model.step(catch_dist=0, desired_speed=3, fov_deg=fov)
        return np.array([math.hypot(a.dx, a.dy) for a in model.all_agents])

    def test_distant_front_accelerates_and_rear_brakes(self):
        for scheme in ("zonal", "voronoi"):
            with self.subTest(model=scheme):
                speeds = self.advance(self.pair(interaction_model=scheme))
                gain = -math.expm1(-.15) / .15
                response = .5 * 3 * math.tanh((80 - 30) / 30) * gain
                np.testing.assert_allclose(speeds, [3 + response, 3 - response])

    def test_close_neighbour_reverses_distance_response(self):
        speeds = self.advance(self.pair(separation=5))
        self.assertLess(speeds[0], 3)
        self.assertGreater(speeds[1], 3)

    def test_zero_social_speed_is_fixed_speed(self):
        for distance in (5, 30, 80):
            np.testing.assert_allclose(self.advance(self.pair(separation=distance, social_speed=0)), [3, 3])

    def test_stronger_social_speed_produces_stronger_response(self):
        weak = self.advance(self.pair(social_speed=.2))
        strong = self.advance(self.pair(social_speed=.8))
        self.assertGreater(strong[0], weak[0])
        self.assertLess(strong[1], weak[1])

    def test_preferred_spacing_has_no_distance_force(self):
        np.testing.assert_allclose(self.advance(self.pair(separation=30)), [3, 3])

    def test_sideways_spacing_force_has_no_longitudinal_component(self):
        model = self.pair()
        model.all_agents[1].x, model.all_agents[1].y = 400, 580
        np.testing.assert_allclose(self.advance(model), [3, 3], atol=1e-12)

    def test_rear_blind_spot_and_interaction_cutoff(self):
        self.assertEqual(self.advance(self.pair(), fov=270)[1], 3)
        np.testing.assert_allclose(self.advance(self.pair(separation=200)), [3, 3])

    def test_periodic_distance_matches_unwrapped_pair(self):
        reference = self.advance(self.pair(separation=50))
        model = self.pair()
        model.env.bounded = False
        model.all_agents[0].x, model.all_agents[1].x = 995, 45
        np.testing.assert_allclose(self.advance(model), reference)

    def test_speed_recovers_without_social_neighbours(self):
        model = self.pair(separation=200, speed_relaxation=.4)
        for agent in model.all_agents:
            agent.dx = 5
        np.testing.assert_allclose(self.advance(model), np.full(2, 3 + 2 * math.exp(-.4)))

    def test_individual_social_response_overrides_type(self):
        model = self.pair()
        kind = model.agent_types["A1"]
        kind.set_distribution("social_speed", dist="uniform", low=0, high=1, color_by=False)
        for agent, response in zip(model.all_agents, (0, 1)):
            agent.individual["social_speed"] = response
        speeds = self.advance(model)
        self.assertEqual(speeds[0], 3)
        self.assertLess(speeds[1], 3)

    def test_temporal_noise_is_independent_and_reproducible(self):
        def series():
            model = self.pair(separation=200, social_speed=0, speed_noise=.3)
            return np.array([self.advance(model) for _ in range(40)])
        first, second = series(), series()
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.std(first[:, 0]), .1)
        self.assertFalse(np.allclose(first[:, 0], first[:, 1]))
        self.assertTrue(np.all((first >= 0) & (first <= 9)))

    def test_zero_recovery_and_speed_ceiling(self):
        speeds = self.advance(self.pair(social_speed=1, speed_relaxation=0, max_speed_scale=1.1))
        self.assertLessEqual(speeds.max(), 3.3 + 1e-12)
        self.assertGreaterEqual(speeds.min(), 0)

    def test_stopped_agent_preserves_heading(self):
        model = self.pair(separation=200, speed_relaxation=0)
        model.agent_types["A1"].max_turn = .6
        for agent in model.all_agents:
            agent.dx, agent.dy, agent.heading = 0., 0., math.pi / 2
        np.testing.assert_allclose(self.advance(model), [0, 0])
        for agent in model.all_agents:
            self.assertAlmostEqual(agent.heading, math.pi / 2)

    def test_configuration_and_distributions_round_trip(self):
        kind = self.pair().agent_types["A1"]
        for field in ("social_speed", "preferred_distance", "speed_relaxation", "speed_noise", "max_speed_scale"):
            kind.set_distribution(field, dist="normal", color_by=False)
        restored = AgentType.from_dict("A1", kind.to_dict())
        self.assertEqual(kind.to_dict(), restored.to_dict())
        restored.validate()

    def test_invalid_parameters_rejected(self):
        for parameters in ({"preferred_distance": 0}, {"speed_noise": -1},
                           {"speed_relaxation": float("nan")}, {"max_speed_scale": .9}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.pair(**parameters)


class SpatialGroupTests(unittest.TestCase):
    def test_multiplier_merges_nearby_components(self):
        agents = [SimpleNamespace(x=x, y=20.) for x in (10, 11, 15, 16)]
        arena = Environment(100, 100)
        self.assertEqual(spatial_groups(agents, arena, multiplier=2.5)["groups"], 2)
        self.assertEqual(spatial_groups(agents, arena, multiplier=5)["groups"], 1)

    def test_four_corners_are_connected_in_periodic_space(self):
        agents = [SimpleNamespace(x=x, y=y) for x, y in ((1, 1), (99, 1), (1, 99), (99, 99))]
        result = spatial_groups(agents, Environment(100, 100, bounded=False))
        self.assertEqual((result["groups"], result["isolated"]), (1, 0))

    def test_invalid_multiplier_rejected(self):
        for multiplier in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                spatial_groups([], multiplier=multiplier)


if __name__ == "__main__":
    unittest.main()
