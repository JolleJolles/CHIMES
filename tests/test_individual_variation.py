"""Individual trait sampling, live application and colour consistency."""
import unittest

import numpy as np

from chimesim.agents import AgentType, _values_to_colors
from chimesim.model import Model


class IndividualVariationTests(unittest.TestCase):
    def test_overlapping_radius_distribution_restarts_reliably(self):
        for seed in range(10):
            kind = AgentType("A1", r_r=3, r_o=90, r_a=132.5)
            kind.set_distribution("r_a", mean=132.5, std=20,
                                  clip=(65, 300), truncate=True)
            model = Model([kind], seed=seed)
            model.init_agents("A1", 200)
            values = np.array([a.individual["r_a"] for a in model.all_agents])
            self.assertTrue(np.all((values >= 90) & (values <= 300)))
            kind.apply_variation(model.all_agents, "r_a", rng=model.rng)
            self.assertTrue(all(a.individual["r_a"] >= 90 for a in model.all_agents))

    def test_impossible_radius_distribution_fails_without_assignment(self):
        model, kind = self.model()
        before = [dict(a.individual) for a in model.all_agents]
        kind.set_distribution("r_a", dist="uniform", low=0, high=1)
        with self.assertRaisesRegex(ValueError, "Cannot sample ordered zone radii"):
            kind.apply_variation(model.all_agents, "r_a", rng=model.rng)
        self.assertEqual(before, [a.individual for a in model.all_agents])

    def model(self):
        kind = AgentType("A1", n_default=40)
        model = Model([kind], seed=15)
        model.init_agents("A1", 40)
        return model, kind

    def test_bounded_normal_has_no_endpoint_pileups(self):
        kind = AgentType("A1")
        kind.set_distribution("speed", mean=1, std=.8, clip=(.8, 1.2), truncate=True)
        values = kind._sample(2000, rng=np.random.default_rng(12))["speed_scale"]
        self.assertTrue(np.all((values > .8) & (values < 1.2)))
        self.assertGreater(values.std(), .08)

    def test_optional_bounds_roundtrip_as_standard_json(self):
        import json
        kind = AgentType("A1")
        kind.set_distribution("speed", mean=1, std=.2, clip=(0, None), truncate=True)
        encoded = json.dumps(kind.to_dict(), allow_nan=False)
        restored = AgentType.from_dict("A1", json.loads(encoded))
        self.assertTrue(np.all(restored._sample(100)["speed_scale"] >= 0))

    def test_legacy_clip_semantics_remain_available(self):
        kind = AgentType("A1")
        kind.set_distribution("speed", mean=1, std=.8, clip=(.8, 1.2))
        values = kind._sample(2000, rng=np.random.default_rng(12))["speed_scale"]
        self.assertGreater(np.count_nonzero(values == .8), 100)
        self.assertGreater(np.count_nonzero(values == 1.2), 100)

    def test_bounded_normal_configuration_roundtrip_and_seed(self):
        kind = AgentType("A1")
        kind.set_distribution("social_speed", mean=.5, std=.3, clip=(0, 1), truncate=True)
        restored = AgentType.from_dict("A1", kind.to_dict())
        self.assertEqual(restored.to_dict(), kind.to_dict())
        np.testing.assert_array_equal(kind._sample(50, np.random.default_rng(5))["social_speed"],
                                      restored._sample(50, np.random.default_rng(5))["social_speed"])

    def test_apply_keeps_positions_and_other_traits(self):
        model, kind = self.model()
        kind.set_distribution("r_a", dist="uniform", low=60, high=100, color_by=False)
        kind.apply_variation(model.all_agents, "r_a", rng=model.rng)
        old_radii = [a.individual["r_a"] for a in model.all_agents]
        positions = [(a.x, a.y, a.dx, a.dy, a.heading) for a in model.all_agents]
        kind.set_distribution("speed_scale", dist="uniform", low=.5, high=1.5, color_by=False)
        kind.apply_variation(model.all_agents, "speed_scale", rng=model.rng)
        self.assertEqual(old_radii, [a.individual["r_a"] for a in model.all_agents])
        self.assertEqual(positions, [(a.x, a.y, a.dx, a.dy, a.heading) for a in model.all_agents])
        self.assertGreater(np.std([a.individual["speed_scale"] for a in model.all_agents]), .1)

    def test_fixed_clears_only_the_selected_trait(self):
        model, kind = self.model()
        kind.set_distribution("speed_scale", dist="uniform", low=.5, high=1.5, color_by=False)
        kind.set_distribution("r_a", dist="uniform", low=60, high=100, color_by=False)
        kind.assign_individual(model.all_agents, rng=model.rng)
        radii = [a.individual["r_a"] for a in model.all_agents]
        del kind._distributions["speed_scale"]
        kind.speed_scale = 1.2
        kind.apply_variation(model.all_agents, "speed_scale", rng=model.rng)
        self.assertTrue(all("speed_scale" not in a.individual for a in model.all_agents))
        self.assertEqual(radii, [a.individual["r_a"] for a in model.all_agents])

    def test_bad_zone_samples_leave_all_individuals_unchanged(self):
        model, kind = self.model()
        old = [a.individual.copy() for a in model.all_agents]
        kind.set_distribution("r_a", dist="uniform", low=0, high=5, color_by=False)
        with self.assertRaises(ValueError):
            kind.apply_variation(model.all_agents, "r_a", rng=model.rng)
        self.assertEqual(old, [a.individual for a in model.all_agents])

    def test_two_fixed_radii_can_be_changed_atomically(self):
        model, kind = self.model()
        kind.r_r, kind.r_o = 40, 50
        kind.apply_variation(model.all_agents, ["r_r", "r_o"], rng=model.rng)
        self.assertTrue(all(not a.individual for a in model.all_agents))

    def test_colour_selection_does_not_resample_traits(self):
        model, kind = self.model()
        kind.set_distribution("speed_scale", dist="uniform", low=.5, high=1.5, color_by=False)
        kind.apply_variation(model.all_agents, "speed_scale", rng=model.rng)
        previous = [a.individual.copy() for a in model.all_agents]
        kind.color_by = "speed_scale"
        kind.update_colors(model.all_agents)
        self.assertGreater(len({a.color for a in model.all_agents}), 1)
        kind.color_by = "flee_dist"
        kind.update_colors(model.all_agents)
        self.assertEqual(len({a.color for a in model.all_agents}), 1)
        self.assertEqual(len(set(kind._colorbar)), 1)
        self.assertEqual(previous, [a.individual for a in model.all_agents])
        kind.color_by = None
        kind.update_colors(model.all_agents)
        self.assertTrue(all(a.color is None for a in model.all_agents))

    def test_new_agents_use_the_same_colour_scale(self):
        model, kind = self.model()
        kind.set_distribution("speed_scale", dist="uniform", low=.5, high=1.5, color_by=True)
        kind.apply_variation(model.all_agents, "speed_scale", rng=model.rng)
        model.init_agents("A1", 20)
        values = [a.individual["speed_scale"] for a in model.all_agents]
        self.assertEqual([a.color for a in model.all_agents], _values_to_colors(values, kind.colormap))

    def test_zero_spread_normal_is_constant(self):
        kind = AgentType("A1")
        kind.set_distribution("speed_scale", mean=1, std=0, clip=(.5, 1.5), truncate=True)
        np.testing.assert_array_equal(kind._sample(10)["speed_scale"], np.ones(10))


if __name__ == "__main__":
    unittest.main()
