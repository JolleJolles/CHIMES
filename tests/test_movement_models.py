"""Movement rules must differ mechanistically, not just by their menu names."""
import math
import json
from pathlib import Path
import os
import unittest
from unittest.mock import patch
import numpy as np
import chimesim
from chimesim.agents import AgentType
from chimesim.environment import Environment
from chimesim.model import Model
from chimesim.model_registry import MODELS, model_id


class MovementModelsTests(unittest.TestCase):
    def pair(self, kind, **parameters):
        t = AgentType('A1', interaction_model=kind, r_r=0, r_o=60, r_a=100,
                      max_turn=math.pi, noise=0, **parameters)
        m = Model([t], seed=4)
        m.env = Environment(500, 500, bounded=False)
        m.init_agents('A1', 2)
        first, other = m.all_agents
        first.x, first.y, first.dx, first.dy, first.heading = 100, 100, 1, 0, 0
        other.x, other.y, other.dx, other.dy, other.heading = 120, 100, 0, 1, math.pi / 2
        return m, first, other

    def test_overlapping_zones_include_same_neighbour_twice(self):
        angles = []
        for kind in ('couzin2002', 'zonal_overlap', 'couzin2005'):
            m, a, _ = self.pair(kind)
            m.step(desired_speed=1, fov_deg=360)
            angles.append(a.heading)
        # Exclusive zones align; overlap adds approach; 2005 also includes own heading.
        np.testing.assert_allclose(angles, [math.pi / 2, math.pi / 4, math.atan2(1, 2)])

    def test_overlap_can_be_removed_by_moving_inner_radius(self):
        m, a, _ = self.pair('zonal_overlap', attraction_start=40)
        m.step(desired_speed=1, fov_deg=360)
        self.assertAlmostEqual(a.heading, math.pi / 2)

    def test_disjoint_overlap_matches_couzin_when_boundaries_are_equal(self):
        states=[]
        for kind in ('couzin2002','zonal_overlap'):
            m,a,b=self.pair(kind, attraction_start=60)
            # A rear neighbour inside avoidance should be perceived in both,
            # even with a narrower main FOV and an irrelevant zero visual cutoff.
            m.agent_types['A1'].r_r=30
            b.x=80
            m.step(desired_speed=1,fov_deg=120,visual_range=0)
            states.append((a.x,a.y,a.dx,a.dy))
        np.testing.assert_allclose(*states)

    def test_shared_zone_does_not_depend_on_hidden_orientation_radius(self):
        results = []
        for radius in (0, 1000):
            m, a, _ = self.pair('couzin2005')
            m.agent_types['A1'].r_o = radius
            m.agent_types['A1'].validate()
            m.step(desired_speed=1, fov_deg=360)
            results.append((a.x, a.y, a.dx, a.dy))
        np.testing.assert_allclose(*results)

    def test_repulsion_still_has_priority_in_both_overlap_models(self):
        for kind in ('zonal_overlap', 'couzin2005'):
            m, a, _ = self.pair(kind)
            t = m.agent_types['A1']; t.r_r = 30; t.attraction_start = 30
            m.step(desired_speed=1, fov_deg=360)
            self.assertAlmostEqual(abs(a.heading), math.pi)

    def test_vicsek_circular_average_includes_self_and_moves_on_old_heading(self):
        m, a, _ = self.pair('vicsek')
        m.step(desired_speed=2, fov_deg=0, visual_range=0)
        self.assertAlmostEqual(a.heading, math.pi / 4)
        self.assertAlmostEqual(a.x, 102)
        self.assertAlmostEqual(a.y, 100)
        self.assertAlmostEqual(math.hypot(a.dx, a.dy), 2)

    def test_vicsek_uniform_noise_has_specified_full_width(self):
        m, a, _ = self.pair('vicsek')
        m.groups['A1'].pop()
        t=m.agent_types['A1'];t.noise=.8;t.max_turn=0;t.heading_noise_memory=100;t.social_speed=1;t.speed_noise=1
        turns=[]
        for _ in range(1200):
            previous=a.heading
            m.step(desired_speed=1)
            turns.append((a.heading-previous+math.pi)%(2*math.pi)-math.pi)
        self.assertLessEqual(max(abs(v) for v in turns), .4)
        self.assertAlmostEqual(np.std(turns), .8/math.sqrt(12), delta=.015)
        self.assertAlmostEqual(math.hypot(a.dx,a.dy),1)

    def test_vicsek_periodic_neighbours(self):
        m,a,b=self.pair('vicsek')
        a.x,b.x=1,499
        m.agent_types['A1'].r_a=5
        m.step(desired_speed=1)
        self.assertAlmostEqual(a.heading,math.pi/4)

    def test_restricted_couzin_ignores_speed_memory_and_detection_extensions(self):
        for kind in ('couzin2002','couzin2005'):
            m,a,_=self.pair(kind, social_speed=1, speed_noise=1, heading_noise_memory=100)
            m.step(desired_speed=2, fov_deg=360, visual_range=0)
            self.assertGreater(a.dy,0)
            self.assertAlmostEqual(math.hypot(a.dx,a.dy),2)
            self.assertIsNone(a.heading_error)

    def test_boids_matches_documented_reference_rules(self):
        fixture=json.loads((Path(__file__).parent/'fixtures'/'boids_reference.json').read_text())
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                t=AgentType('A1', interaction_model='boids',
                            r_a=case.get('perception',55), separation_weight=.35,
                            alignment_weight=1, cohesion_weight=2, max_acceleration=0)
                m=Model([t],seed=2)
                m.env=Environment(1000,1000,bounded=False)
                m.init_agents('A1',len(case['initial']))
                for agent,state in zip(m.all_agents,case['initial']):
                    agent.x,agent.y,agent.dx,agent.dy=state
                    agent.heading=math.atan2(agent.dy,agent.dx)
                for _ in range(case['steps']):
                    m.step(desired_speed=4.5,fov_deg=0,visual_range=0,
                           min_speed=0 if case['name']=='exact_overlap_stopped' else 2.025)
                actual=[[a.x,a.y,a.dx,a.dy] for a in m.all_agents]
                if case['name']=='isolated_minimum_speed':
                    # Compare moving-agent reference dynamics; with an explicit
                    # positive floor CHIMES also starts a stationary agent along
                    # its retained heading, rather than keeping speed at zero.
                    actual=actual[:1]
                    expected=case['expected'][:1]
                else:
                    expected=case['expected']
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)

    def test_boids_minimum_speed_is_explicit_and_defaults_to_zero(self):
        for floor,maximum,expected in ((0,4.5,.1),(1.5,4.5,1.5),(1.5,6,1.5)):
            m,a,b=self.pair('boids',separation_weight=0,alignment_weight=0,cohesion_weight=0)
            a.dx,a.dy=.1,0
            m.step(desired_speed=maximum,min_speed=floor)
            self.assertAlmostEqual(math.hypot(a.dx,a.dy),expected)
        m,a,b=self.pair('boids',separation_weight=0,alignment_weight=0,cohesion_weight=0)
        a.dx,a.dy,a.heading=0,0,math.pi/2
        m.step(desired_speed=4,min_speed=1)
        self.assertAlmostEqual(a.dx,0,places=10)
        self.assertAlmostEqual(a.dy,1)
        with self.assertRaisesRegex(ValueError,'min_speed'):
            m.step(desired_speed=1,min_speed=2)

    def test_boids_schooling_preset_keeps_spacing_and_local_alignment(self):
        from chimesim import metrics
        t=AgentType('A1',interaction_model='boids',r_a=55,separation_weight=.35,
                    alignment_weight=1,cohesion_weight=2)
        m=Model([t],seed=42)
        m.env=Environment(1000,1000,bounded=False)
        m.init_agents('A1',160)
        for _ in range(800):
            m.step(desired_speed=4.5,min_speed=2.025)
        self.assertGreater(metrics.mean_nnd(m.all_agents,m.env),5)
        p=np.array([[a.x,a.y] for a in m.all_agents])
        v=np.array([[a.dx,a.dy] for a in m.all_agents])
        headings=v/np.linalg.norm(v,axis=1)[:,None]
        delta=p[:,None]-p[None,:]
        delta-=np.round(delta/1000)*1000
        distance=np.linalg.norm(delta,axis=2)
        neighbours=(distance>0)&(distance<55)
        self.assertTrue(neighbours.any())
        self.assertGreater(np.mean((headings@headings.T)[neighbours]),.7)

    def test_boids_legacy_separation_radius_does_not_change_neighbourhood(self):
        results=[]
        for radius in (0,1000):
            m,a,b=self.pair('boids')
            m.agent_types['A1'].r_r=radius
            m.agent_types['A1'].validate()
            m.step(desired_speed=3)
            results.append([a.dx,a.dy])
        np.testing.assert_allclose(*results)

    def test_all_models_round_trip_and_generate_seeded_datasets(self):
        for kind in MODELS:
            with self.subTest(model=kind):
                scenario=chimesim.settings(source='defaults', model=kind, environment={'bounded':False,'width':300,'height':200},
                                          agents={'A1':{'n':4}})
                restored=chimesim.settings(source=scenario)
                self.assertEqual(restored,scenario)
                one=chimesim.simulate(settings=scenario,n_steps=3,burn_in=0,seed=9,save_trajectories=True,verbose=False)
                two=chimesim.simulate(settings=scenario,n_steps=3,burn_in=0,seed=9,save_trajectories=True,verbose=False)
                self.assertEqual(one['trajectories'],two['trajectories'])
                self.assertEqual(len(one['trajectories']),12)

    def test_legacy_settings_keep_extensions(self):
        t=AgentType.from_dict('A1',{'interaction_model':'zonal','social_speed':.5})
        self.assertFalse(t.restricted)
        self.assertEqual(model_id(t.interaction_model),'couzin2002')

    def test_restricted_vicsek_rejects_walls_but_defaults_to_periodic(self):
        scenario=chimesim.settings(source='defaults',model='vicsek')
        self.assertFalse(scenario['environment']['bounded'])
        with self.assertRaisesRegex(ValueError,'periodic'):
            chimesim.settings(source=scenario,environment={'bounded':True})

    def test_shared_zone_distributions_do_not_require_hidden_radius_order(self):
        t=AgentType('A1',interaction_model='couzin2005',r_r=40,r_o=0,r_a=100)
        t.set_distribution('r_r',dist='uniform',low=45,high=55)
        t.constrain_zones(200)
        m=Model([t],seed=2);m.init_agents('A1',20)
        self.assertTrue(all(45 <= a.individual['r_r'] <= 55 for a in m.all_agents))


@unittest.skipUnless(os.environ.get('CHIMESIM_GUI_TESTS')=='1','Requires a graphical display')
class ModelSelectorTests(unittest.TestCase):
    def test_interactive_switches_models_and_rebuilds_relevant_controls(self):
        import tkinter as tk
        from tkinter import ttk
        import chimesim.ui_tk as ui
        errors=[];saved=[];real_tk=tk.Tk
        def children(w):
            for child in w.winfo_children():
                yield child
                yield from children(child)
        def factory(*args,**kwargs):
            root=real_tk(*args,**kwargs)
            root.report_callback_exception=lambda kind,value,tb:(errors.append(value),root.quit())
            kinds=iter(MODELS)
            def choose_next():
                try:
                    kind=next(kinds,None)
                    widgets=list(children(root))
                    if kind is None:
                        next(w for w in widgets if isinstance(w,tk.Button) and w.cget('text')=='Exit').invoke()
                        return
                    selector=next(w for w in widgets if isinstance(w,ttk.Combobox) and MODELS['vicsek'] in w.cget('values'))
                    selector.set(MODELS[kind]);selector.event_generate('<<ComboboxSelected>>')
                    root.after(120,check)
                except BaseException as exc:
                    errors.append(exc);root.quit()
            def check():
                try:
                    current=saved[-1]['agents']['A1']
                    self.assertTrue(current['restricted'])
                    widgets=list(children(root))
                    labels=[str(w.cget('text')) for w in widgets if isinstance(w,tk.Label)]
                    if current['interaction_model'] == 'boids':
                        toggle=next(w for w in widgets if isinstance(w,tk.Checkbutton) and w.cget('text')=='Show perception circles')
                        toggle.invoke()
                        self.assertTrue(any(w.find_withtag('perception-circles') for w in widgets if isinstance(w,tk.Canvas)))
                        radius_labels=[w for w in widgets if isinstance(w,tk.Label) and
                                       ((w.cget('text')=='Perception radius') or
                                        (w.cget('textvariable') and w.getvar(w.cget('textvariable'))=='Perception radius'))]
                        self.assertEqual(sum(w.master.winfo_manager()=='pack' for w in radius_labels),1)
                        for title in ('Visual range', 'Capture distance'):
                            label=next(w for w in widgets if isinstance(w,tk.Label) and w.cget('text')==title)
                            self.assertEqual(label.master.winfo_manager(),'')
                        boundary=next(w for w in widgets if isinstance(w,tk.Checkbutton) and w.cget('text')=='Bounded arena')
                        self.assertEqual(boundary.winfo_manager(),'')
                        self.assertFalse(any(' · mean' in label for label in labels))
                        label=next(w for w in widgets if isinstance(w,tk.Label)
                                   and w.cget('textvariable') and w.getvar(w.cget('textvariable'))=='Maximum speed')
                        speed=next(w for w in children(label.master) if isinstance(w,tk.Scale))
                        self.assertAlmostEqual(float(speed.get()),3.)
                        steering=[w for w in widgets if isinstance(w,tk.Label) and w.cget('textvariable')
                                  and w.getvar(w.cget('textvariable'))=='Steering force limit']
                        self.assertFalse(steering)
                    if current['interaction_model']=='vicsek':
                        self.assertFalse(saved[-1]['bounded'])
                        self.assertFalse(any('Social speed' in label for label in labels))
                    root.after(40,choose_next)
                except BaseException as exc:
                    errors.append(exc);root.quit()
            root.after(250,choose_next)
            root.after(12000,lambda:(errors.append(TimeoutError('Model switching timed out')),root.quit()))
            return root
        with patch.object(ui.tk,'Tk',factory),patch.object(ui,'_save_settings',saved.append),patch.object(ui,'display_preferences',return_value={}):
            ui.interactive(settings={'width':400,'height':300,'bounded':False,'agents':{'A1':{'n':5}}})
        if errors: raise errors[0]

    def test_dataset_gui_model_choices_round_trip(self):
        from chimesim.ui_simulation import SimulationSetup
        app=SimulationSetup()
        try:
            for kind in MODELS:
                scenario=chimesim.settings(source='defaults',model=kind,environment={'bounded':False},agents={'A1':{'n':3}})
                app.apply_settings(scenario)
                config=app.configuration()
                self.assertEqual(config['agents']['A1']['interaction_model'],kind)
                self.assertTrue(config['agents']['A1']['restricted'])
                if kind == 'boids':
                    for key in ('visual_range','capture_distance','wall_avoid_radius','wall_strength','thigmotaxis','mask'):
                        self.assertEqual(app.entries[key].master.winfo_manager(),'')
                    self.assertEqual(app.boundary_check.winfo_manager(),'')
                    self.assertEqual(app.mask_button.winfo_manager(),'')
                    self.assertAlmostEqual(config['agents']['A1']['speed_scale'],1.)
        finally:
            app.destroy()
