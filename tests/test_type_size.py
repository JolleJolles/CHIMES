"""Type-specific silhouettes affect perception, but leave movement unchanged."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from chimesim import settings, simulate
from chimesim.agents import AgentType, Boid
from chimesim.body_geometry import body_size, body_geometry
from chimesim.environment import Environment
from chimesim.perception import perceive
from chimesim.ui_visual_network import VisualNetwork


class TypeSizeTests(unittest.TestCase):
    def test_full_body_and_circle_scale_by_type(self):
        t=AgentType('A1',size=12,size_multiplier=2)
        self.assertEqual(body_size(t),24)
        self.assertEqual(body_size(SimpleNamespace(size=12)),12)
        circle=body_geometry(0,0,0,body_size(t),'Circle with tail')['circle']
        self.assertEqual(circle,(-8,-8,8,8))
        polygon=body_geometry(0,0,0,body_size(t))['polygon']
        self.assertEqual(max(polygon[::2])-min(polygon[::2]),24)

    def test_movement_is_identical_and_effective_traits_are_saved(self):
        scenario=settings(source='defaults',agents={'A1':{'n':5,'size_multiplier':1},'A2':{'n':2}})
        larger=settings(source=scenario,agents={**scenario['agents'],'A1':{**scenario['agents']['A1'],'size_multiplier':3}})
        a=simulate(settings=scenario,n_steps=4,burn_in=0,seed=12,save_trajectories=True,verbose=False)
        b=simulate(settings=larger,n_steps=4,burn_in=0,seed=12,save_trajectories=True,verbose=False)
        self.assertEqual(a['trajectories'],b['trajectories'])
        self.assertEqual({r['size_multiplier'] for r in b['individuals'] if r['agent_type']=='A1'},{3})
        self.assertEqual(settings(source=larger),larger)
        with self.assertRaises(ValueError):
            settings(source='defaults',agents={'A1':{'size_multiplier':-1}})

    def test_larger_body_has_more_angular_extent(self):
        a=Boid(0,20,50,1,0,AgentType('A1',size=6))
        b=Boid(1,80,50,1,0,AgentType('A2',size=6))
        first=perceive([a,b],body_style='Circle with tail')
        b.type.size_multiplier=3
        second=perceive([a,b],body_style='Circle with tail')
        self.assertGreater(second.angular_extent[0,1],first.angular_extent[0,1]*2)

    def test_paused_visual_network_recomputes_after_size_change(self):
        canvas=Mock()
        network=VisualNetwork(canvas,lambda:None)
        scene=[Boid(0,20,50,1,0,AgentType('A1',size=6)),Boid(1,80,50,1,0,AgentType('A2',size=6))]
        env=Environment(200,100)
        def draw():network.draw(scene,env,(1,0,0),True,'Circle with tail',270,100,0,0)
        try:
            draw();network.pending.result(timeout=3);draw()
            original=network.result.angular_extent[0,1]
            scene[1].type.size_multiplier=3
            draw()
            self.assertIsNotNone(network.pending)
            network.pending.result(timeout=3);draw()
            self.assertGreater(network.result.angular_extent[0,1],original*2)
        finally:
            network.close()
