"""Directed steering, opt-in capture and migration of old saved scenarios."""
import math
import unittest
import numpy as np
from chimesim import settings, simulate
from chimesim.agents import AgentType, Boid
from chimesim.environment import Environment
from chimesim.model import Model


class RelationshipTests(unittest.TestCase):
    def model(self, relationship='pursues', *, bounded=True, reverse=False, wall=False):
        first=AgentType('A1',noise=0,max_turn=.1,r_r=0,r_o=0,r_a=100,flee_dist=100,
                        **{relationship:['A2']})
        other=AgentType('A2',noise=0,max_turn=0)
        model=Model([other,first] if reverse else [first,other],seed=3)
        model.env=Environment(200,200,bounded=bounded,avoid_strength=0,
                              wall_shapes=[{'kind':'Rectangle','points':[[.55,0],[.6,1]]}] if wall else [])
        model.groups['A1']=[Boid(0,100,100,0,3,first)]
        model.groups['A2']=[Boid(1,130,100,0,3,other)]
        return model

    def test_pursuit_and_fleeing_turn_oppositely_within_limit(self):
        for relationship, sign in [('pursues',1),('flees',-1)]:
            m=self.model(relationship)
            m.step(10,3,fov_deg=360,visual_range=100)
            a=m.groups['A1'][0]
            self.assertGreater(a.dx*sign,0)
            self.assertAlmostEqual(abs(a.heading-math.pi/2),.1)
            self.assertAlmostEqual(math.hypot(a.dx,a.dy),3)
            self.assertEqual(a.state,'alarm' if relationship=='flees' else 'calm')

    def test_fov_range_multiplier_and_walls_apply_to_both_relationships(self):
        for relationship in ('pursues','flees'):
            for reason in ('fov','range','multiplier','wall'):
                with self.subTest(relationship=relationship,reason=reason):
                    m=self.model(relationship,wall=reason=='wall')
                    if reason=='multiplier':m.agent_types['A1'].visual_range_multiplier=0
                    m.step(0,3,fov_deg=30 if reason=='fov' else 360,
                           visual_range=0 if reason=='range' else 100)
                    self.assertAlmostEqual(m.groups['A1'][0].dx,0)
                    self.assertEqual(m.groups['A1'][0].state,'calm')

    def test_pursuit_never_removes_targets_without_capture(self):
        m=self.model();m.groups['A2'][0].x=101
        m.step(40,3,fov_deg=360)
        self.assertEqual(len(m.all_agents),2)

    def test_capture_is_explicit_distance_limited_and_independent_of_vision(self):
        for distance, expected in [(0,2),(10,2),(40,1)]:
            m=self.model('captures')
            m.step(distance,3,fov_deg=0,visual_range=0)
            self.assertEqual(len(m.all_agents),expected)
        m=self.model('captures',wall=True)
        m.step(40,3)
        self.assertEqual(len(m.all_agents),2)

    def test_periodic_capture_and_steering_use_shortest_distance(self):
        for relationship in ('captures','pursues','flees'):
            m=self.model(relationship,bounded=False)
            m.groups['A1'][0].x=2;m.groups['A2'][0].x=198
            m.step(10,3,fov_deg=360,visual_range=10)
            if relationship=='captures':self.assertEqual(len(m.all_agents),1)
            else:self.assertGreater(m.groups['A1'][0].dx*(-1 if relationship=='pursues' else 1),0)

    def test_shared_snapshot_makes_type_order_irrelevant_and_mutual_capture_simultaneous(self):
        states=[]
        for reverse in (False,True):
            m=self.model(reverse=reverse)
            m.step(0,3,fov_deg=360)
            states.append([(a.x,a.y,a.dx,a.dy) for name in sorted(m.groups) for a in m.groups[name]])
        self.assertEqual(states[0],states[1])
        m=self.model('captures');m.agent_types['A2'].captures.add('A1')
        m.step(40,3)
        self.assertEqual(len(m.all_agents),0)

    def test_default_model_has_one_generic_type_and_no_capture(self):
        m=Model()
        self.assertEqual(list(m.agent_types),['A1'])
        self.assertFalse(m.agent_types['A1'].captures)

    def test_relationship_validation(self):
        for values in ({'pursues':'A2'},{'pursues':['missing']},{'captures':['A1']},
                       {'pursues':['A2'],'flees':['A2']}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                settings(source='defaults',agents={'A1':values,'A2':{}})

    def test_old_settings_migrate_to_nonlethal_pursuit_and_new_settings_roundtrip(self):
        scenario=settings(source={'catch_dist':40,'agents':{'A1':{'n':1,'hunts':['A2']},'A2':{'n':1}}})
        self.assertEqual(scenario['movement']['capture_distance'],40)
        self.assertNotIn('catch_dist',scenario['movement'])
        self.assertEqual(scenario['agents']['A1']['pursues'],['A2'])
        self.assertEqual(scenario['agents']['A1']['captures'],[])
        self.assertNotIn('hunts',scenario['agents']['A1'])
        self.assertEqual(settings(source=scenario),scenario)
        out=simulate(settings=scenario,n_steps=2,burn_in=0,seed=3,verbose=False)
        self.assertEqual(out['summary'][-1]['A2_n'],1)

    def test_capture_exports_keep_initial_individuals_but_end_trajectories(self):
        from unittest.mock import patch
        from chimesim.agents import Boid
        def initialise(model,name,n):
            kind=model.agent_types[name]
            model.groups[name]=[Boid(0 if name=='A1' else 1,100,100,1,0,kind)]
        scenario=settings(source='defaults',agents={'A1':{'n':1,'captures':['A2']},'A2':{'n':1}})
        with patch.object(Model,'init_agents',initialise):
            result=simulate(settings=scenario,n_steps=2,burn_in=0,save_trajectories=True,seed=2,verbose=False)
        self.assertEqual(len(result['individuals']),2)
        self.assertEqual({row['id'] for row in result['trajectories']},{0})
        self.assertEqual(result['summary'][0]['A2_n'],0)
