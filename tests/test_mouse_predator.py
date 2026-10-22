"""External mouse threats share perception but never become captured agents."""
import unittest
import math
from chimesim.agents import AgentType, Boid
from chimesim.model import Model
from chimesim.environment import Environment
from chimesim.ui_events import arena_pointer


class MousePredatorTests(unittest.TestCase):
    def model(self):
        t=AgentType('A1',noise=0,max_turn=.1)
        m=Model([t]);m.env=Environment(200,200,avoid_strength=0)
        m.groups['A1']=[Boid(0,100,100,0,3,t)]
        return m

    def test_external_threat_needs_no_type_relationship_and_never_captures(self):
        m=self.model();m.step(100,3,fov_deg=360,external_threats=[(110,100)])
        a=m.all_agents[0]
        self.assertEqual(len(m.all_agents),1)
        self.assertLess(a.dx,0)
        self.assertEqual(a.state,'alarm')
        self.assertAlmostEqual(a.heading,math.pi/2+.1)
        self.assertEqual(list(m.groups),['A1'])
        m.step(100,3,fov_deg=360)
        self.assertEqual(a.state,'calm')

    def test_external_threat_respects_fov_and_range(self):
        for fov,radius in [(30,100),(360,0)]:
            m=self.model();m.step(0,3,fov_deg=fov,visual_range=radius,external_threats=[(110,100)])
            self.assertAlmostEqual(m.all_agents[0].dx,0)
            self.assertEqual(m.all_agents[0].state,'calm')
        with self.assertRaises(ValueError):self.model().step(external_threats=[(float('nan'),2)])

    def test_pointer_transforms_and_ignores_letterbox_and_outside_viewport(self):
        self.assertEqual(arena_pointer((50,70),(2,10,30),(200,100),(300,200)),(20,20))
        self.assertEqual(arena_pointer((50,70),(2,-10,-30),(200,100),(300,200)),(30,50))
        self.assertIsNone(arena_pointer(None,(1,0,0),(100,100),(200,200)))
        self.assertIsNone(arena_pointer((5,5),(1,10,10),(100,100),(200,200)))
        self.assertIsNone(arena_pointer((150,50),(1,0,0),(100,100),(200,200)))
        self.assertIsNone(arena_pointer((210,50),(1,0,0),(400,100),(200,200)))
