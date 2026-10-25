import os
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock, patch

import numpy as np

from chimesim.agents import AgentType, Boid
from chimesim.environment import Environment
from chimesim import metrics
from chimesim.ui_positions import PositionMeans, PositionHistory, calculate_position_frame, _palette, _indices


class PositionDisplayTests(unittest.TestCase):
    def frame(self, step, front, distance):
        return dict(step=step, scores=dict(id=np.array([7]),
                    front_back_relative=np.array([front]), centroid_distance_relative=np.array([distance])))

    def test_means_exclude_missing_samples_and_replace_duplicate_step(self):
        means = PositionMeans()
        means.add(self.frame(1, -.5, .2))
        result = means.add(self.frame(2, np.nan, np.nan))
        self.assertAlmostEqual(result['front_back_relative'][0], -.5)
        self.assertEqual(means.totals[(7, 'front_back_relative')][1], 1)
        result = means.add(self.frame(3, .5, .8))
        self.assertAlmostEqual(result['front_back_relative'][0], 0)
        self.assertAlmostEqual(result['centroid_distance_relative'][0], .5)
        result = means.add(self.frame(3, 1., .6))
        self.assertAlmostEqual(result['front_back_relative'][0], .25)
        self.assertEqual(means.totals[(7, 'front_back_relative')][1], 2)

    def test_largest_group_excludes_opposed_second_group_and_isolates(self):
        t = AgentType('A1')
        agents = [Boid(i, 10+i, 10, 1 if i<3 else -1, 0, t) for i in range(6)]
        labels = np.array([0,0,0,1,1,-1])
        result = metrics.largest_group_metrics(agents, Environment(100,100,bounded=True), labels)
        self.assertEqual(result['A1'], (1.,1.))
        self.assertEqual(metrics.polarisation(agents), 0.)
        missing = metrics.largest_group_metrics(agents, None, np.full(6,-1))
        self.assertTrue(np.isnan(missing['A1']).all())

    def test_palette_is_identical_for_agents_and_history(self):
        obj = PositionHistory.__new__(PositionHistory)
        obj.colour_limits = {'Spatial leadership':[-1.,1.]}
        obj.palette_var = Mock(get=lambda: 'plasma')
        obj.colour_var = Mock(get=lambda: 'Spatial leadership')
        obj.statistic_var = Mock(get=lambda: 'Instantaneous')
        obj.latest_frame = self.frame(1, .3, .2)
        obj.latest = obj.latest_frame['scores']
        index = _indices([.3], 'Spatial leadership')[0]
        expected = '#%02x%02x%02x' % tuple(_palette('Spatial leadership','plasma')[index])
        self.assertEqual(obj.colours(), {7:expected})

    def test_poll_does_not_schedule_twice_when_ready_callback_starts_next_job(self):
        obj = PositionHistory.__new__(PositionHistory)
        obj.closed = False
        obj.pending = None
        obj.window = Mock()
        obj._consume = Mock(return_value=True)
        def ready():
            obj.pending = ('next', Mock())
            obj.poll_id = 'already-scheduled'
        obj.on_ready = ready
        obj._poll()
        obj.window.after.assert_not_called()

    def test_position_work_is_single_pending_job_on_detached_snapshot(self):
        obj = PositionHistory.__new__(PositionHistory)
        obj.executor = ThreadPoolExecutor(max_workers=1)
        obj.pending = None
        obj.generation = 0
        obj.closed = False
        obj.signature = None
        obj.last_time = -float('inf')
        obj.trait_labels = {'r_a':'Perception radius'}
        obj.window = Mock()
        gate = Event()
        original = calculate_position_frame
        def delayed(*args):
            gate.wait(timeout=3)
            return original(*args)
        t = AgentType('A1')
        agent = Boid(1,10,10,1,0,t)
        try:
            with patch('chimesim.ui_positions.calculate_position_frame', delayed):
                obj.update([agent], Environment(100,100), 2.5, 1, force=True)
                future = obj.pending[1]
                self.assertFalse(future.done())
                obj.update([agent], Environment(100,100), 2.5, 2, force=True)
                self.assertIs(obj.pending[1], future)
                agent.x = 90
                t.r_a = 500
                gate.set()
                frame = future.result(timeout=3)
                self.assertEqual(frame['step'], 1)
                self.assertEqual(frame['traits'][0,0], 75)
        finally:
            gate.set()
            obj.close()


@unittest.skipUnless(os.environ.get('CHIMESIM_GUI_TESTS') == '1', 'Requires a graphical display')
class PositionDisplayGUITests(unittest.TestCase):
    def test_editable_controls_shared_colours_and_background_history(self):
        import tkinter as tk
        import chimesim.ui_tk as ui
        errors, histories, models = [], [], []
        real_tk, real_history, real_prepare = tk.Tk, ui.PositionHistory, ui._prepare_model

        def children(widget):
            for child in widget.winfo_children():
                yield child
                yield from children(child)

        def history(*args, **kwargs):
            obj = real_history(*args, **kwargs)
            histories.append(obj)
            return obj

        def prepare(*args, **kwargs):
            result = real_prepare(*args, **kwargs)
            models.append(result[1])
            return result

        def factory(*args, **kwargs):
            root = real_tk(*args, **kwargs)
            root.report_callback_exception = lambda kind, value, tb: (errors.append(value), root.quit())

            def exercise():
                try:
                    widgets = list(children(root))
                    radius_label = next(w for w in widgets if isinstance(w,tk.Label) and w.cget('text')=='Perception radius')
                    scale = next(w for w in children(radius_label.master) if isinstance(w,tk.Scale))
                    entry = scale.value_entry
                    entry.delete(0,tk.END)
                    entry.insert(0,'123.456')
                    entry.commit_value()
                    self.assertAlmostEqual(models[0].agent_types['A1'].r_a, 123.456)
                    self.assertEqual(entry.get(),'123.456')
                    minimum_label=next(w for w in widgets if isinstance(w,tk.Label) and w.cget('text')=='Minimum speed')
                    minimum_scale=next(w for w in children(minimum_label.master) if isinstance(w,tk.Scale))
                    self.assertEqual(minimum_scale.value_entry.get(),'0')
                    minimum_scale.value_entry.delete(0,tk.END)
                    minimum_scale.value_entry.insert(0,'1.25')
                    minimum_scale.value_entry.commit_value()
                    self.assertEqual(minimum_scale.value_entry.get(),'1.25')
                    # Out-of-range edits restore the current value.
                    entry.delete(0,tk.END)
                    entry.insert(0,'nan')
                    entry.commit_value()
                    self.assertEqual(entry.get(),'123.456')
                    checks = {w.cget('text'):w for w in widgets if isinstance(w,tk.Checkbutton)}
                    variation_header=next(w for w in widgets if isinstance(w,tk.Label) and w.cget('text')=='Individual variation')
                    self.assertEqual(variation_header.master.winfo_manager(),'')
                    checks['Individual variation'].invoke()
                    self.assertEqual(variation_header.master.winfo_manager(),'pack')
                    checks['Individual variation'].invoke()
                    self.assertEqual(variation_header.master.winfo_manager(),'')
                    buttons={w.cget('text'):w for w in widgets if isinstance(w,tk.Button)}
                    buttons['Attract'].invoke()
                    self.assertEqual(str(buttons['Attract'].cget('relief')),'sunken')
                    self.assertEqual(str(buttons['Repel'].cget('relief')),'raised')
                    buttons['Off'].invoke()
                    checks['Show individual positions'].invoke()
                    checks['Show perception circles'].invoke()
                    next(w for w in widgets if isinstance(w,tk.Button) and w.cget('text')=='Start').invoke()
                    root.after(1000, inspect)
                except BaseException as exc:
                    errors.append(exc)
                    root.quit()

            def inspect():
                try:
                    h = histories[0]
                    self.assertTrue(h.frames)
                    self.assertGreaterEqual(min(np.hypot(a.dx,a.dy) for a in models[0].all_agents),1.25-1e-9)
                    h.palette_var.set('plasma')
                    h.statistic_var.set('Mean since reset')
                    h.change_view()
                    widgets = list(children(root))
                    next(w for w in widgets if isinstance(w,tk.Button) and w.cget('text')=='Pause').invoke()
                    next(w for w in widgets if isinstance(w,tk.Button) and w.cget('text')=='A1 colour').invoke()
                    self.assertEqual(models[0].agent_types['A1'].color,'#123456')
                    self.assertEqual(h.colour_var.get(),'Type colours')
                    for name,kind in models[0].agent_types.items():
                        title=next(w for w in children(root) if isinstance(w,tk.Label) and w.cget('text')==f'Agent type {name[1:]}')
                        self.assertEqual(title.cget('fg'),kind.color)
                    canvas = next(w for w in widgets if isinstance(w,tk.Canvas) and w.master is root)
                    self.assertTrue(canvas.find_withtag('perception-circles'))
                    base_colours={t.color for t in models[0].agent_types.values()}
                    self.assertEqual(len(base_colours),3)
                    self.assertTrue(set(canvas.itemcget(item,'fill') for item in canvas.find_withtag('agents')) <= base_colours)
                    h.change_view()
                    expected = set(h.colours().values())
                    self.assertTrue(expected)
                    self.assertTrue(set(canvas.itemcget(item,'fill') for item in canvas.find_withtag('agents')) <= expected)
                    self.assertTrue(h.canvas.find_withtag('trait-swatch'))
                    checks={w.cget('text'):w for w in children(root) if isinstance(w,tk.Checkbutton)}
                    checks['A1'].invoke()
                    checks['A1'].invoke()
                    self.assertEqual(models[0].agent_types['A1'].color,'#123456')
                    reset=next(w for w in children(root) if isinstance(w,tk.Button) and w.cget('text')=='Reset type colours')
                    reset.invoke()
                    from chimesim.agents import TYPE_COLOURS
                    self.assertEqual({name:t.color for name,t in models[0].agent_types.items()},TYPE_COLOURS)
                    next(w for w in children(root) if isinstance(w,tk.Button) and w.cget('text')=='Exit').invoke()
                except BaseException as exc:
                    errors.append(exc)
                    root.quit()

            root.after(250,exercise)
            root.after(10000,lambda: (errors.append(TimeoutError('Display GUI check timed out')),root.quit()))
            return root

        with patch.object(ui.tk,'Tk',factory), patch.object(ui,'PositionHistory',history), \
             patch.object(ui,'_prepare_model',prepare), patch.object(ui,'_save_settings'), \
             patch.object(ui,'display_preferences',return_value={}), \
             patch.object(ui.colorchooser,'askcolor',return_value=((18,52,86),'#123456')):
            ui.interactive(settings={'interaction_model':'boids','bounded':False,'width':300,'height':200,
                                     'spawn_radius':20,'agents':{name:{'n_default':6,'interaction_model':'boids'} for name in ('A1','A2','A3')}})
        if errors:
            raise errors[0]
