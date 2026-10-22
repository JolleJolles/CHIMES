"""Repulsion perception, saved settings and compact free-space spawning."""
import math
import unittest
from unittest.mock import patch
import numpy as np
from chimesim import settings, simulate
from chimesim.agents import AgentType, Boid
from chimesim.model import Model
from chimesim.environment import Environment


class RepulsionTests(unittest.TestCase):
    def model(self, kind='zonal'):
        kind_type=AgentType('A1',noise=0,max_turn=math.pi,r_r=5,r_o=6,r_a=10,interaction_model=kind)
        model=Model([kind_type],seed=4)
        model.env=Environment(100,100,bounded=False,avoid_strength=0)
        model.groups['A1']=[Boid(0,50,50,1,0,kind_type),Boid(1,48,48,1,0,kind_type)]
        return model

    def test_omni_repulsion_sees_behind_without_widening_alignment(self):
        for kind in ('zonal','voronoi'):
            narrow,omni=self.model(kind),self.model(kind)
            narrow.step(0,1,fov_deg=90)
            omni.step(0,1,fov_deg=90,repulsion_fov_deg=360)
            self.assertAlmostEqual(narrow.all_agents[0].dy,0)
            self.assertGreater(omni.all_agents[0].dy,.5)

    def test_omni_still_respects_walls_and_detection_range(self):
        for distance in (0,10):
            model=self.model()
            with patch.object(model.env,'los_mask',return_value=np.zeros((2,2),dtype=bool)):
                model.step(0,1,fov_deg=90,repulsion_fov_deg=360,visual_range=distance)
            self.assertAlmostEqual(model.all_agents[0].dy,0)

    def test_zero_repulsion_angle_disables_repulsion(self):
        model=self.model()
        model.step(0,1,fov_deg=1,repulsion_fov_deg=0)
        self.assertAlmostEqual(model.all_agents[0].dy,0)

    def test_settings_roundtrip_and_validation(self):
        scenario=settings(source='defaults',movement={'repulsion_fov_deg':360},environment={'spawn_radius':30},agents={'A1':{'n':4}})
        result=simulate(settings=scenario,n_steps=1,burn_in=0,seed=2,verbose=False)
        self.assertEqual(result['settings']['movement']['repulsion_fov_deg'],360)
        self.assertEqual(result['settings']['environment']['spawn_radius'],30)
        with self.assertRaises(ValueError):settings(movement={'repulsion_fov_deg':361})
        with self.assertRaises(ValueError):settings(environment={'spawn_radius':-1})

    def test_compact_positions_are_inside_circle_and_free(self):
        env=Environment(200,100,bounded=False,spawn_radius=20)
        points=env.sample_positions(100,np.random.default_rng(4))
        self.assertTrue((np.linalg.norm(points-[100,50],axis=1)<=20).all())
        self.assertEqual(env.sample_positions(0,np.random.default_rng(1)).shape,(0,2))

    def test_default_reproduces_explicit_matching_fov(self):
        a,b=self.model(),self.model()
        for _ in range(3):
            a.step(0,1,fov_deg=270)
            b.step(0,1,fov_deg=270,repulsion_fov_deg=270)
        self.assertEqual([(x.x,x.y,x.dx,x.dy) for x in a.all_agents],[(x.x,x.y,x.dx,x.dy) for x in b.all_agents])
