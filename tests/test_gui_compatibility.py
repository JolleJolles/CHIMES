"""Opt-in real-Tk compatibility checks (use Xvfb on headless Linux)."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('CHIMESIM_GUI_TESTS') == '1', 'Set CHIMESIM_GUI_TESTS=1 with a graphical display')
class GUICompatibilityTests(unittest.TestCase):
    def test_session_recording_settings_visual_network_and_reopen(self):
        import tkinter as tk
        import chimesim.ui_tk as ui
        real_tk = tk.Tk
        errors, saved = [], []

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)

            def factory(*args, **kwargs):
                root = real_tk(*args, **kwargs)
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
                        root.after(1200, finish)
                    except BaseException as exc:
                        errors.append(exc)
                        root.quit()

                def finish():
                    try:
                        widgets = list(descendants(root))
                        buttons = {w.cget('text'): w for w in widgets if isinstance(w, tk.Button)}
                        canvas = next(w for w in widgets if isinstance(w, tk.Canvas) and w.master is root)
                        buttons['Pause'].invoke()
                        self.assertTrue(canvas.find_withtag('agents'))
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
                 patch.object(ui.filedialog, 'asksaveasfilename', filename):
                for _ in range(2):
                    ui.run(settings={'width': 300, 'height': 200, 'seed': 12,
                                     'agents': {'A1': {'n_default': 8}},
                                     'wall_shapes': [{'kind': 'Rectangle', 'points': [[.45, .1], [.55, .35]]}]})
                    if errors:
                        raise errors[0]
            self.assertTrue(saved)
