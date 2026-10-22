"""Batch output equivalence, progress and interruption checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from chimesim import simulate
from chimesim.experiment import run_experiment


class SimulationSetupTests(unittest.TestCase):
    def settings(self):
        return dict(n_steps=4,burn_in=2,n_reps=2,seed=12,save_trajectories=True,
                    agents={'A1':{'n_default':5,'distributions':{'speed':{'dist':'uniform','low':.8,'high':1.2}}}})

    def test_streamed_results_match_seeded_api(self):
        settings=self.settings()
        reference=simulate(settings=settings,verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            events=[]
            result=run_experiment(settings,directory,progress=events.append)
            folder=Path(result['output'])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['completed_reps'],2)
            self.assertEqual(result['completed_steps'],12)
            rows=list(csv.DictReader((folder/'summary.csv').read_text().splitlines()))
            self.assertEqual(len(rows),len(reference['summary']))
            for actual,expected in zip(rows,reference['summary']):
                for key,value in expected.items():
                    if isinstance(value, str): self.assertEqual(actual[key], value)
                    else: self.assertAlmostEqual(float(actual[key]),value)
            rows=list(csv.DictReader((folder/'trajectories.csv').read_text().splitlines()))
            self.assertEqual(len(rows),len(reference['trajectories']))
            for actual,expected in zip(rows,reference['trajectories']):
                for key,value in expected.items():
                    if isinstance(value,str):self.assertEqual(actual[key],value)
                    else:self.assertAlmostEqual(float(actual[key]),value)
            self.assertEqual(events[0]['completed'],0)
            self.assertEqual(events[-1]['completed'],events[-1]['total'])
            self.assertEqual(json.loads((folder/'status.json').read_text())['status'],'complete')

    def test_cancel_keeps_partial_repeat_and_settings(self):
        settings=self.settings();settings['n_steps']=10
        rows=[]
        result=simulate(settings=settings,verbose=False,on_row=lambda kind,row:rows.append((kind,row)),
                        should_cancel=lambda:len([r for kind,r in rows if kind=='summary'])>=3,
                        collect_results=False)
        self.assertTrue(result['cancelled'])
        self.assertEqual(result['completed_reps'],0)
        self.assertEqual(result['completed_steps'],5)
        self.assertEqual(result['summary'],[])
        self.assertEqual(len([r for kind,r in rows if kind=='summary']),3)

    def test_cancel_before_start_creates_preserved_status(self):
        with tempfile.TemporaryDirectory() as directory:
            result=run_experiment(self.settings(),directory,should_cancel=lambda:True)
            self.assertEqual(result['status'],'cancelled')
            self.assertEqual(result['completed_steps'],0)
            self.assertTrue((Path(result['output'])/'settings.json').exists())

    def test_failure_preserves_status_and_output_is_never_overwritten(self):
        settings=self.settings();settings['mask']='does-not-exist.png'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):run_experiment(settings,directory)
            folder=next(Path(directory).iterdir())
            self.assertEqual(json.loads((folder/'status.json').read_text())['status'],'failed')
            a=run_experiment(self.settings(),directory)
            b=run_experiment(self.settings(),directory)
            self.assertNotEqual(a['output'],b['output'])
