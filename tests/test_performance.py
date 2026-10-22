"""Performance counters and rates without running a scaling benchmark."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chimesim import settings, simulate
from chimesim.performance import workload


class PerformanceTests(unittest.TestCase):
    def scenario(self):
        return settings(source="defaults", agents={"A1": {"n": 4}})

    def test_single_run_counts_burn_in_and_repeats_without_changing_results(self):
        kwargs = dict(settings=self.scenario(), n_steps=3, burn_in=2, n_reps=2, seed=4,
                      save_trajectories=True, verbose=False)
        reference = simulate(**kwargs)
        measured = simulate(**kwargs, performance=True)
        self.assertNotIn('performance', reference)
        self.assertEqual(measured['summary'], reference['summary'])
        self.assertEqual(measured['trajectories'], reference['trajectories'])
        report = measured['performance']
        self.assertEqual(report['completed_steps'], 10)
        self.assertEqual(report['agent_updates'], 40)
        self.assertAlmostEqual(report['steps_per_minute'], 600/report['elapsed_seconds'])
        self.assertEqual([r['completed_steps'] for r in report['conditions'][0]['repeats']], [5,5])
        self.assertEqual([r['seed'] for r in report['conditions'][0]['repeats']], [4,5])
        self.assertIn('total_memory_bytes', report['system'])
        self.assertIn('logical_cpu_count', report['system'])
        json.dumps(report, allow_nan=False)

    def test_grid_reports_each_population_separately(self):
        result = simulate(settings=self.scenario(), conditions={'agents.A1.n': [2,5]},
                          n_steps=2, burn_in=1, n_reps=2, seed=4, performance=True, verbose=False)
        report = result['performance']
        self.assertEqual(report['completed_steps'], 12)
        self.assertEqual(report['agent_updates'], 42)
        self.assertEqual([r['initial_agents'] for r in report['conditions']], [2,5])
        self.assertEqual([r['agent_updates'] for r in report['conditions']], [12,30])
        self.assertEqual(report['conditions'][1]['values'], {'agents.A1.n':5})

    def test_cancel_during_burn_in_counts_only_completed_steps(self):
        checks = iter([False, False, True])
        result = simulate(settings=self.scenario(), n_steps=3, burn_in=3, seed=4,
                          performance=True, should_cancel=lambda: next(checks), verbose=False)
        condition = result['performance']['conditions'][0]
        self.assertTrue(result['cancelled'])
        self.assertEqual(condition['agent_updates'], 4)
        self.assertEqual(condition['completed_steps'], 1)
        self.assertTrue(condition['repeats'][0]['cancelled'])
        stopped = simulate(settings=self.scenario(), n_steps=1, burn_in=0, seed=4,
                           performance=True, should_cancel=lambda: True, verbose=False)
        self.assertEqual(stopped['performance']['agent_updates'], 0)
        self.assertEqual(stopped['performance']['steps_per_minute'], 0)

    def test_streaming_writes_report_for_condition_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            result = simulate(settings=self.scenario(), n_steps=1, n_reps=1, burn_in=0,
                              conditions={'agents.A1.n':[2,3]}, output=directory, performance=True, seed=4)
            saved = json.loads((Path(result['output'])/'performance.json').read_text())
            self.assertEqual(saved, result['performance'])
            self.assertEqual(saved['completed_steps'], 2)
            self.assertEqual(saved['agent_updates'], 5)

    def test_report_scope_and_invalid_flag(self):
        self.assertEqual(workload(2., 1., 10, 100)['steps_per_minute'], 300.)
        with self.assertRaises(TypeError):
            simulate(settings=self.scenario(), performance='yes')
        with patch('chimesim.experiment.launch') as launch:
            simulate(settings=self.scenario(), gui=True, performance=True)
        self.assertTrue(launch.call_args.kwargs['performance'])
