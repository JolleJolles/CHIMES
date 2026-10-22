"""Detection cutoffs, constrained heterogeneous radii and persistent errors."""
import unittest
import numpy as np
from chimesim import simulate
from chimesim.agents import AgentType
from chimesim.model import Model
from chimesim.environment import Environment


class DirectionalityTests(unittest.TestCase):
    def test_zero_detection_removes_social_hunting_and_fleeing_steering(self):
        a = AgentType('A1', noise=0, pursues={'A2'})
        b = AgentType('A2', noise=0)
        model = Model([a, b], seed=7)
        model.env = Environment(400, 400, bounded=False)
        model.init_agents('A1', 1); model.init_agents('A2', 1)
        first, other = model.all_agents
        first.x, first.y, first.dx, first.dy, first.heading = 100, 100, 0, 3, np.pi / 2
        other.x, other.y, other.dx, other.dy = 110, 100, 3, 0
        model.step(0, 3, fov_deg=360, visual_range=0)
        self.assertAlmostEqual(first.dx, 0)
        self.assertAlmostEqual(first.dy, 3)
        self.assertEqual(first.state, 'calm')

    def test_visual_cutoff_and_multiplier_limit_direction_response(self):
        for multiplier, expected_turn in [(0, False), (.5, False), (2, True)]:
            t = AgentType('A1', noise=0, r_r=0, r_o=0, r_a=100, visual_range_multiplier=multiplier)
            m = Model([t], seed=12);m.env = Environment(400, 400, bounded=False);m.init_agents('A1', 2)
            first, other = m.all_agents
            first.x, first.y, first.dx, first.dy = 100, 100, 0, 3
            other.x, other.y = 130, 100
            m.step(0, 3, fov_deg=360, visual_range=20)
            self.assertEqual(abs(first.dx) > .01, expected_turn)

    def test_range_constraints_preserve_order_and_saveable_zero_distributions(self):
        t = AgentType('A1')
        t.set_distribution('r_a', mean=75, std=15, clip=(20, 150), truncate=True)
        m = Model([t], seed=2);m.init_agents('A1', 30)
        t.constrain_zones(20, m.all_agents)
        self.assertLessEqual(t.r_r, t.r_o);self.assertLessEqual(t.r_o, t.r_a);self.assertLessEqual(t.r_a, 20)
        for a in m.all_agents:
            radii = [a.individual.get(k, getattr(t, k)) for k in ('r_r','r_o','r_a')]
            self.assertTrue(0 <= radii[0] <= radii[1] <= radii[2] <= 20)
        t.constrain_zones(0, m.all_agents)
        restored = AgentType.from_dict('A1', t.to_dict())
        m2 = Model([restored], seed=2);m2.init_agents('A1', 30)
        self.assertTrue(all(a.individual['r_a'] == 0 for a in m2.all_agents))

    def test_correlated_errors_retain_stationary_spread_with_persistent_turns(self):
        correlations = []
        for memory in (0, 10):
            t = AgentType('A1', noise=.03, max_turn=np.pi, heading_noise_memory=memory)
            m = Model([t], seed=12);m.env = Environment(400, 400, bounded=False);m.init_agents('A1', 1)
            a = m.all_agents[0];turns=[]
            for _ in range(1800):
                old = a.heading;m.step(0, 1)
                turns.append((a.heading - old + np.pi) % (2*np.pi) - np.pi)
            turns=np.array(turns)
            correlations.append(np.corrcoef(turns[:-1],turns[1:])[0,1])
            self.assertLess(abs(turns.std() - .03), .006)
        self.assertLess(abs(correlations[0]), .1)
        self.assertGreater(correlations[1], .8)

    def test_headless_settings_preserve_detection_and_noise_memory(self):
        out = simulate(settings={'visual_range':0, 'agents':{'A1':{'n_default':2, 'heading_noise_memory':5}}},
                       n_steps=2, burn_in=0, verbose=False)
        self.assertEqual(out['settings']['movement']['visual_range'], 0)
        self.assertEqual(out['settings']['agents']['A1']['heading_noise_memory'],5)
