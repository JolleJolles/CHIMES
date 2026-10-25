import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from chimesim.agents import AgentType
from chimesim.environment import Environment
from chimesim.ui_perception import draw_perception_circles


class PerceptionCirclesTests(unittest.TestCase):
    def canvas(self):
        canvas = Mock()
        canvas.winfo_width.return_value = 500
        canvas.winfo_height.return_value = 500
        canvas.winfo_rgb.return_value = (0x5656, 0xb1b1, 0xfcfc)
        return canvas

    def test_uses_individual_radius_and_view_transform(self):
        canvas = self.canvas()
        t = AgentType('A1', interaction_model='boids', r_a=75)
        agent = SimpleNamespace(type=t, individual={'r_a': 20}, x=100, y=100)
        draw_perception_circles(canvas, [agent], Environment(500,500,bounded=True),
                                (2,10,15), True, 1)
        args, options = canvas.create_oval.call_args
        self.assertEqual(args, (170,175,250,255))
        self.assertEqual(options['width'], .5)
        self.assertGreaterEqual(min(int(options['fill'][i:i+2],16) for i in (1,3,5)), 250)
        self.assertTrue(options['outline'])

    def test_wraps_at_seam_and_toggle_clears_overlay(self):
        canvas = self.canvas()
        t = AgentType('A1', interaction_model='boids', r_a=20)
        agent = SimpleNamespace(type=t, individual={}, x=5, y=250)
        env = Environment(500,500,bounded=False)
        draw_perception_circles(canvas, [agent], env, (1,0,0), True, 1)
        centres = [(call.args[0]+call.args[2])/2 for call in canvas.create_oval.call_args_list]
        self.assertEqual(centres, [5,505])
        canvas.reset_mock()
        draw_perception_circles(canvas, [agent], env, (1,0,0), False, 1)
        canvas.delete.assert_called_once_with('perception-circles')
        canvas.create_oval.assert_not_called()
