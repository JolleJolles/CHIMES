"""Shared scenarios, persistence, reproducible grids and streamed outputs."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chimesim
from chimesim.configuration import flatten, save_session, display_preferences
from chimesim.experiment import run_experiment


class ConfigurationTests(unittest.TestCase):
    def scenario(self):
        return chimesim.settings(source="defaults", model="zonal",
                                 environment={"width": 400, "height": 300, "bounded": False},
                                 agents={"A1": {"n": 3, "repulsion": 5, "orientation": 15, "attraction": 40}})

    def test_public_surface_and_scenario_independence(self):
        self.assertEqual(chimesim.__all__, ["settings", "interactive", "simulate"])
        for name in ("Model", "Boid", "BOID", "PREY", "PREDATOR", "Environment", "AgentType"):
            self.assertFalse(hasattr(chimesim, name), name)
        source = self.scenario()
        original = copy.deepcopy(source)
        scenario = chimesim.settings(source=source, movement={"visual_range": 0})
        scenario["agents"]["A1"]["n_default"] = 9
        self.assertEqual(source, original)
        self.assertEqual(scenario["movement"]["visual_range"], 0)
        self.assertNotIn("seed", scenario)

    def test_session_separates_display_and_scenario_and_explicit_settings(self):
        with tempfile.TemporaryDirectory() as directory, patch('chimesim.configuration._STATE_DIRECTORY', Path(directory)):
            flat = flatten(self.scenario())
            flat.update(window_layout={"Simulation": {"x": 10}}, show_scale_bar=False, seed=87)
            save_session(flat)
            self.assertNotIn("window_layout", json.loads((Path(directory)/"scenario.json").read_text()))
            self.assertIn("window_layout", json.loads((Path(directory)/"display.json").read_text()))
            self.assertEqual(display_preferences()["window_layout"], {"Simulation": {"x": 10}})
            self.assertNotIn("width", display_preferences())
            self.assertEqual(chimesim.settings()["environment"]["width"], 400)
            restored = chimesim.simulate(n_steps=1, burn_in=0, seed=2, verbose=False)
            self.assertEqual(restored["settings"]["environment"]["width"], 400)
            independent = chimesim.simulate(settings={"agents": {"test": {"n": 1}}},
                                           n_steps=1, burn_in=0, seed=2, verbose=False)
            self.assertEqual(independent["settings"]["environment"]["width"], 1920)
            self.assertEqual(chimesim.settings(source="defaults")["environment"]["width"], 1920)

    def test_nested_relative_mask_and_legacy_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scenario.json"
            scenario = self.scenario()
            scenario["environment"]["mask"] = "wall.png"
            path.write_text(json.dumps(scenario))
            self.assertEqual(chimesim.settings(source=path)["environment"]["mask"], str((path.parent/"wall.png").resolve()))
            legacy = chimesim.settings(source={"width": 500, "agents": {"fish": {"n": 4}}})
            self.assertEqual(legacy["environment"]["width"], 500)

    def test_generated_seed_is_returned_and_reproduces_run(self):
        result = chimesim.simulate(settings=self.scenario(), n_steps=2, burn_in=0, verbose=False)
        self.assertIsInstance(result["run"]["seed"], int)
        again = chimesim.simulate(settings={"scenario": result["settings"], "run": result["run"]}, verbose=False)
        self.assertEqual(result["summary"], again["summary"])

    def test_grid_repeats_seeds_rows_and_streaming_agree(self):
        scenario = self.scenario()
        before = copy.deepcopy(scenario)
        conditions = {"agents.A1.speed_scale": [.5, 1, 1.5], "agents.A1.n": [2, 4]}
        events = []
        result = chimesim.simulate(settings=scenario, conditions=conditions, n_steps=2,
                                  n_reps=2, burn_in=0, seed=40, save_trajectories=True,
                                  progress=events.append, verbose=False)
        self.assertEqual(scenario, before)
        self.assertEqual(len(result["conditions"]), 6)
        self.assertEqual(len(result["summary"]), 24)
        self.assertEqual([c["run"]["seed"] for c in result["conditions"]], list(range(40,52,2)))
        self.assertEqual(events[-1]["completed"], events[-1]["total"])
        with tempfile.TemporaryDirectory() as directory:
            written = chimesim.simulate(settings=scenario, conditions=conditions, n_steps=2,
                                       n_reps=2, burn_in=0, seed=40, save_trajectories=True, output=directory)
            folder = Path(written["output"])
            self.assertEqual(written["completed_reps"], 12)
            self.assertEqual(len(json.loads((folder/"conditions.json").read_text())), 6)
            rows = list(csv.DictReader((folder/"summary.csv").read_text().splitlines()))
            self.assertEqual(len(rows),24)
            self.assertEqual([int(r["condition"]) for r in rows], [r["condition"] for r in result["summary"]])
            rerun = chimesim.simulate(settings=folder/"settings.json", verbose=False)
            self.assertEqual(rerun["summary"], result["summary"])

    def test_grid_validates_all_conditions_before_execution(self):
        with self.assertRaisesRegex(ValueError, "Unknown condition"):
            chimesim.simulate(settings=self.scenario(), conditions={"agents.A1.typo": [1]}, verbose=False)
        with self.assertRaises(ValueError):
            chimesim.simulate(settings=self.scenario(), conditions={"agents.A1.n": [1,-1]}, verbose=False)
        with self.assertRaises(ValueError):
            chimesim.settings(environment={"typo": 1})

    def test_grid_cancel_stops_before_later_conditions(self):
        rows = []
        result = chimesim.simulate(settings=self.scenario(), conditions={"agents.A1.n": [2,3]},
                                  n_steps=5, n_reps=2, burn_in=0, seed=3, verbose=False,
                                  on_row=lambda kind,row: rows.append(row) if kind=="summary" else None, should_cancel=lambda: len(rows)>=2)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["completed_steps"], 2)
        self.assertEqual({r["condition"] for r in rows}, {0})
        self.assertEqual(len(result["conditions"]), 2)

    def test_gui_receives_shared_scenario_and_grid(self):
        with patch("chimesim.experiment.launch") as launch:
            chimesim.simulate(settings=self.scenario(), gui=True, conditions={"agents.A1.n": [2,3]})
        self.assertEqual(launch.call_args.kwargs["conditions"], {"agents.A1.n": [2,3]})
