"""Tracking-style exports, individual assignments and online repeat means."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from chimesim import settings, simulate


class TrackingOutputTests(unittest.TestCase):
    def scenario(self):
        return settings(source='defaults', agents={
            'A1': {'n': 3, 'speed_scale': 1., 'distributions': {
                'speed_scale': {'dist': 'uniform', 'low': .7, 'high': 1.3},
                'social_speed': {'dist': 'uniform', 'low': .1, 'high': .4}}}})

    def test_trajectories_link_to_static_individuals_and_frame_clock(self):
        result = simulate(settings=self.scenario(), n_steps=3, n_reps=2, burn_in=4, seed=4,
                          save_trajectories=True, verbose=False)
        self.assertEqual(len(result['individuals']),6)
        self.assertEqual(len(result['trajectories']),18)
        assignments = {(r['set'],r['id']): r for r in result['individuals']}
        for row in result['trajectories']:
            individual = assignments[row['set'],row['id']]
            self.assertEqual(individual['agent_type'],'A1')
            self.assertNotIn('speed_scale', row)
            self.assertNotIn('social_speed', row)
            self.assertIn('speed_scale', individual)
            self.assertIn('social_speed', individual)
            self.assertEqual(row['frame'], row['t'])
            self.assertAlmostEqual(row['speed'], (row['vx']**2+row['vy']**2)**.5)
        self.assertEqual(sorted({r['frame'] for r in result['trajectories']}), [5,6,7])
        self.assertEqual(len({(r['set'],r['id'],r['frame']) for r in result['trajectories']}),18)

    def test_repeat_means_match_timeseries_and_work_without_raw_tracks(self):
        result = simulate(settings=self.scenario(), n_steps=4, n_reps=2, burn_in=3, seed=3, verbose=False)
        self.assertNotIn('trajectories', result)
        self.assertEqual(len(result['individuals']),6)
        self.assertEqual(len(result['repeats']),2)
        for repeat in result['repeats']:
            rows = [r for r in result['summary'] if r['rep']==repeat['rep']]
            self.assertEqual(repeat['recorded_steps'],4)
            for key in ['A1_polarisation','A1_nnd','A1_speed','A1_n','overall_polarisation']:
                self.assertAlmostEqual(repeat[key+'_mean'],sum(r[key] for r in rows)/4)

    def test_disk_and_memory_outputs_match_and_each_call_gets_own_folder(self):
        kwargs = dict(settings=self.scenario(), n_steps=2, n_reps=2, burn_in=1, seed=2,
                      conditions={'agents.A1.n':[2,3]}, save_trajectories=True, verbose=False)
        memory = simulate(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            first = simulate(**kwargs, output_dir=directory)
            second = simulate(**kwargs, output_dir=directory)
            self.assertNotEqual(first['output_dir'],second['output_dir'])
            folder=Path(first['output_dir'])
            self.assertEqual(folder.parent,Path(directory).resolve())
            self.assertTrue((folder/'data_format.json').exists())
            for kind in ['summary','trajectories','individuals','repeats']:
                disk = list(csv.DictReader((folder/(kind+'.csv')).read_text().splitlines()))
                self.assertEqual(len(disk),len(memory[kind]))
                for actual, expected in zip(disk,memory[kind]):
                    for key, value in expected.items():
                        if isinstance(value,bool):self.assertEqual(actual[key],str(value))
                        elif isinstance(value,str):self.assertEqual(actual[key],value)
                        else:self.assertAlmostEqual(float(actual[key]),value)
            self.assertEqual(len(memory['individuals']),10)
            self.assertEqual(len(memory['trajectories']),20)

    def test_partial_repeat_summary_uses_only_recorded_rows(self):
        rows=[]
        result = simulate(settings=self.scenario(), n_steps=10, burn_in=0, seed=2, verbose=False,
                          on_row=lambda kind,row: rows.append(row) if kind=='summary' else None,
                          should_cancel=lambda:len(rows)>=2)
        self.assertTrue(result['repeats'][0]['cancelled'])
        self.assertEqual(result['repeats'][0]['recorded_steps'],2)
        self.assertAlmostEqual(result['repeats'][0]['A1_speed_mean'],sum(r['A1_speed'] for r in rows)/2)

    def test_cancel_before_recording_has_missing_means_and_ambiguous_paths_rejected(self):
        checks=iter([False,True])
        result = simulate(settings=self.scenario(), burn_in=3, seed=2, verbose=False,
                          should_cancel=lambda:next(checks))
        self.assertEqual(result['repeats'][0]['recorded_steps'],0)
        self.assertIsNone(result['repeats'][0]['A1_speed_mean'])
        with self.assertRaises(ValueError):
            simulate(settings=self.scenario(), output='one', output_dir='two')
