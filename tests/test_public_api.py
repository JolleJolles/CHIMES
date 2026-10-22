"""Public entry-point names, lazy GUI dispatch and configuration precedence."""
import copy
import unittest
from unittest.mock import patch
import chimesim
from chimesim.agents import BOID


class PublicAPITests(unittest.TestCase):
    def test_run_removed_and_interactive_exported(self):
        self.assertFalse(hasattr(chimesim,'run'))
        self.assertNotIn('run',chimesim.__all__)
        self.assertTrue(callable(chimesim.interactive))

    def test_gui_dispatch_prefills_resolved_overrides_without_mutating_settings(self):
        settings={'n_reps':2,'seed':5,'agents':{'fish':{'n_default':8}}}
        before=copy.deepcopy(settings)
        with patch('chimesim.experiment.launch',return_value=None) as launch:
            result=chimesim.simulate(gui=True,settings=settings,n_reps=7,seed=42)
        self.assertIsNone(result)
        resolved=launch.call_args.kwargs['settings']
        self.assertEqual(resolved['n_reps'],7)
        self.assertEqual(resolved['seed'],42)
        self.assertEqual(resolved['agents']['fish']['n_default'],8)
        self.assertEqual(settings,before)

    def test_gui_returns_saved_run_and_console_does_not_exit_with_a_dictionary(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        from chimesim import experiment
        saved={'output_dir':'/example/run', 'status':'complete'}
        app=Mock(result=saved)
        module=SimpleNamespace(SimulationSetup=Mock(return_value=app))
        with patch.dict(sys.modules, {'chimesim.ui_simulation':module}):
            result=chimesim.simulate(gui=True,settings={'agents':{'A1':{'n':2}}})
        self.assertEqual(result,saved)
        app.mainloop.assert_called_once()
        with patch.object(experiment,'launch',return_value=saved):
            self.assertIsNone(experiment.main())

    def test_gui_rejects_direct_worker_callbacks(self):
        with self.assertRaisesRegex(ValueError,'direct'):
            chimesim.simulate(gui=True,progress=lambda _:None)
        with self.assertRaises(TypeError):
            chimesim.simulate(gui='yes')

    def test_direct_mode_still_returns_datasets(self):
        results=chimesim.simulate(n_steps=2,burn_in=0,agent_types=[BOID.clone('fish')],seed=42,verbose=False)
        self.assertEqual(len(results['summary']),2)
        self.assertIn('settings',results)
