"""Opt-in real-Tk compatibility checks (use Xvfb on headless Linux)."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class WindowCleanupTests(unittest.TestCase):
    def test_destruction_drains_events_without_reentering_mainloop(self):
        from unittest.mock import Mock
        try:
            from chimesim.ui_events import destroy_window
            from tkinter import TclError
        except ImportError:
            self.skipTest('Tk is not installed')
        root = Mock()
        root.tk.call.return_value = ('stale', 'pending')
        root.after_cancel.side_effect = [TclError('already cancelled'), None]
        root.tk.dooneevent.side_effect = [1, 1, 0]
        destroy_window(root)
        root.destroy.assert_called_once()
        self.assertEqual(root.after_cancel.call_count, 2)
        self.assertEqual(root.tk.dooneevent.call_count, 3)
        root.mainloop.assert_not_called()


@unittest.skipUnless(os.environ.get('CHIMESIM_GUI_TESTS') == '1', 'Set CHIMESIM_GUI_TESTS=1 with a graphical display')
class GUICompatibilityTests(unittest.TestCase):
    def test_session_recording_settings_visual_network_and_reopen(self):
        import tkinter as tk
        from tkinter import ttk
        import chimesim.ui_tk as ui
        real_tk = tk.Tk
        errors, saved, roots = [], [], []

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)

            def factory(*args, **kwargs):
                root = real_tk(*args, **kwargs)
                roots.append(root)
                root.report_callback_exception = lambda kind, value, tb: (errors.append(value), root.quit())

                def exercise():
                    try:
                        widgets = list(descendants(root))
                        buttons = {w.cget('text'): w for w in widgets if isinstance(w, tk.Button)}
                        checks = {w.cget('text'): w for w in widgets if isinstance(w, tk.Checkbutton)}
                        buttons['Start recording'].invoke()
                        buttons['Start'].invoke()
                        buttons['+'].invoke()
                        buttons['Fit arena'].invoke()
                        checks['Show visual neighbours (directed)'].invoke()
                        checks['Show interaction zones'].invoke()
                        checks['Show information'].invoke()
                        checks['Show individual positions'].invoke()
                        root.after(350, open_selector)
                    except BaseException as exc:
                        errors.append(exc)
                        root.quit()

                def open_selector():
                    try:
                        widgets=list(descendants(root))
                        window=next(w for w in widgets if isinstance(w,tk.Toplevel) and w.title()=='Individual positions')
                        selector=next(w for w in descendants(window) if isinstance(w,ttk.Combobox) and 'All types' in w.cget('values'))
                        selector.tk.call('ttk::combobox::Post',selector._w)
                        root.after(100,lambda: select_agent(selector))
                    except BaseException as exc:
                        errors.append(exc);root.quit()

                def select_agent(selector):
                    try:
                        popup=str(selector.tk.call('ttk::combobox::PopdownWindow',selector._w))
                        listbox=popup+'.f.l'
                        selector.tk.call(listbox,'selection','clear',0,'end')
                        selector.tk.call(listbox,'selection','set',1)
                        selector.tk.call('ttk::combobox::LBSelected',listbox)
                        root.after(150,lambda: check_selection(selector))
                    except BaseException as exc:
                        errors.append(exc);root.quit()

                def check_selection(selector):
                    try:
                        self.assertEqual(selector.get(),'A1')
                        self.assertFalse(ui.popup_active(root))
                        root.after(900,finish)
                    except BaseException as exc:
                        errors.append(exc);root.quit()

                def finish():
                    try:
                        widgets = list(descendants(root))
                        buttons = {w.cget('text'): w for w in widgets if isinstance(w, tk.Button)}
                        canvas = next(w for w in widgets if isinstance(w, tk.Canvas) and w.master is root)
                        buttons['Pause'].invoke()
                        self.assertTrue(canvas.find_withtag('agents'))
                        information=next(w for w in widgets if isinstance(w,tk.Toplevel) and w.title()=='Information')
                        self.assertTrue(information.winfo_ismapped())
                        self.assertGreaterEqual(information.winfo_width(),340)
                        self.assertGreaterEqual(information.winfo_height(),280)
                        info_canvas=next(w for w in descendants(information) if isinstance(w,tk.Canvas))
                        self.assertTrue(info_canvas.find_withtag('milling-title'))
                        self.assertTrue(info_canvas.find_withtag('milling'))
                        self.assertFalse(ui.popup_active(root))
                        zone_window = next(w for w in widgets if isinstance(w, tk.Toplevel) and w.title() == 'Interaction zones')
                        self.assertNotEqual(zone_window.state(), 'withdrawn')
                        zone_canvas = next(w for w in descendants(zone_window) if isinstance(w, tk.Canvas))
                        self.assertTrue(zone_canvas.find_withtag('attraction'))
                        self.assertTrue(zone_canvas.find_withtag('example-agent'))
                        self.assertEqual(zone_canvas.type(zone_canvas.find_withtag('repulsion')[0]), 'oval')
                        buttons['Stop recording'].invoke()
                        buttons['Export recorded data'].invoke()
                        buttons['Save settings'].invoke()
                        self.assertTrue((target / 'trajectory.csv').exists())
                        self.assertTrue((target / 'settings.json').exists())
                        buttons['Exit'].invoke()
                    except BaseException as exc:
                        errors.append(exc)
                        root.quit()

                root.after(300, exercise)
                root.after(15000, lambda: (errors.append(TimeoutError('GUI session timed out')), root.quit()))
                return root

            def filename(**kwargs):
                return str(target / ('settings.json' if 'settings' in kwargs.get('title', '').lower() else 'trajectory.csv'))

            with patch.object(ui.tk, 'Tk', factory), patch.object(ui, '_save_settings', saved.append), \
                 patch.object(ui, 'display_preferences', return_value={}), \
                 patch.object(ui.filedialog, 'asksaveasfilename', filename):
                for _ in range(2):
                    ui.interactive(settings={'width': 300, 'height': 200, 'seed': 12,
                                     'agents': {'A1': {'n_default': 8}}, 'repulsion_fov_deg':360, 'spawn_radius':20,
                                     'window_layout': {'general_movement': {'visible':False},
                                                       'information': {'geometry':'1x1+99999+99999','visible':False},
                                                       'positions': {'geometry':'1x1+99999+99999','visible':False}},
                                     'wall_shapes': [{'kind': 'Rectangle', 'points': [[.45, .1], [.55, .35]]}]})
                    if errors:
                        raise errors[0]
                    root = roots[-1]
                    # Tk's interpreter survives root destruction, as in a kernel.
                    self.assertEqual(root.tk.call('info', 'commands', '.'), '')
                    self.assertIsNone(tk._default_root)
                    self.assertEqual(root.tk.call('after', 'info'), '')
            self.assertTrue(saved)
            self.assertEqual(saved[-1]['repulsion_fov_deg'],360)
            self.assertEqual(saved[-1]['spawn_radius'],20)
            self.assertFalse(saved[-1]['window_layout']['information']['geometry'].startswith('1x1'))
            self.assertFalse(saved[-1]['window_layout']['positions']['geometry'].startswith('1x1'))


@unittest.skipUnless(os.environ.get('CHIMESIM_GUI_TESTS') == '1', 'Set CHIMESIM_GUI_TESTS=1 with a graphical display')
class SimulationSetupGUICompatibilityTests(unittest.TestCase):
    def test_public_simulate_gui_prefills_without_starting(self):
        import chimesim
        from chimesim.ui_simulation import SimulationSetup
        errors = []
        def factory(settings, conditions=None):
            app = SimulationSetup(settings=settings, conditions=conditions)
            def inspect():
                try:
                    self.assertEqual(app.values['n_reps'].get(), '7')
                    self.assertEqual(app.values['seed'].get(), '42')
                    self.assertEqual(app.agent_vars[0]['n_default'].get(), '6')
                    self.assertIsNone(app.worker)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    app.close()
            app.after(200, inspect)
            return app
        with patch('chimesim.ui_simulation.SimulationSetup', factory):
            result = chimesim.simulate(gui=True, settings={'n_reps':2, 'agents':{'fish':{'n_default':6}}},
                                      n_reps=7, seed=42)
        self.assertIsNone(result)
        if errors: raise errors[0]

    def test_separate_runner_completion_settings_and_cancellation(self):
        import json
        from chimesim.ui_simulation import SimulationSetup
        errors = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            app = SimulationSetup()
            app.report_callback_exception = lambda kind, value, tb: (errors.append(value), app.destroy())
            app.values['output'].set(str(target))
            app.values['n_steps'].set('4');app.values['n_reps'].set('2');app.values['burn_in'].set('1')
            app.agent_vars[0]['n_default'].set('5')
            app.performance.set(True)
            app.add_condition('agents.A1.speed_scale', [.8, 1, 1.2])
            app.add_condition()
            from tkinter import ttk
            row=app.condition_body.winfo_children()[-1]
            selector=next(w for w in row.winfo_children() if isinstance(w,ttk.Combobox))
            app.update()
            selector.set('A1 — Attraction radius')
            selector.event_generate('<<ComboboxSelected>>')
            self.assertEqual(app.condition_vars[-1][0].get(),'agents.A1.r_a')
            app.condition_vars[-1][1].set('90, 120, 150')
            app.check_conditions()
            self.assertEqual(app.condition_count.get(),'9 conditions × 2 repeats')
            phase = ['complete']
            def check():
                try:
                    if app.worker is not None:
                        app.after(50, check);return
                    folders = sorted(path for path in target.iterdir() if path.is_dir())
                    data = json.loads((folders[-1] / 'status.json').read_text())
                    if phase[0] == 'complete':
                        self.assertEqual(data['status'], 'complete')
                        self.assertEqual(data['completed_reps'], 18)
                        self.assertTrue((folders[-1]/'performance.json').exists())
                        self.assertEqual(len(json.loads((folders[-1]/'conditions.json').read_text())), 9)
                        self.assertEqual(float(app.bar['value']), 100)
                        self.assertTrue((folders[-1] / 'trajectories.csv').exists())
                        self.assertTrue((folders[-1] / 'individuals.csv').exists())
                        self.assertTrue((folders[-1] / 'repeats.csv').exists())
                        with patch('chimesim.ui_simulation.filedialog.asksaveasfilename', return_value=str(target/'settings.json')):
                            app.save()
                        app.values['width'].set('300')
                        with patch('chimesim.ui_simulation.filedialog.askopenfilename', return_value=str(target/'settings.json')):
                            app.load()
                        self.assertEqual(app.values['width'].get(), '1200')
                        self.assertEqual(len(app.condition_vars), 2)
                        self.assertEqual(Path(app.result['output_dir']),folders[-1].resolve())
                        self.assertEqual(str(app.cancel_button.cget('state')),'disabled')
                        self.assertEqual(str(app.exit_button.cget('state')),'normal')
                        self.assertEqual(app.cancel_button.cget('text'),'Stop simulation')
                        app.values['n_steps'].set('100000');app.values['burn_in'].set('0')
                        app.start();phase[0]='cancelled';app.after(100,app.cancel);app.after(200,check)
                    else:
                        self.assertEqual(data['status'], 'cancelled')
                        self.assertLess(data['completed_steps'], 100000)
                        app.exit_button.invoke()
                except BaseException as exc:
                    errors.append(exc);app.cancel_event.set();app.destroy()
            app.start()
            app.after(200, check)
            app.after(15000, lambda: (errors.append(TimeoutError('Batch GUI timed out')), app.cancel_event.set(), app.destroy()))
            app.mainloop()
            if errors: raise errors[0]
