"""
Headless simulation runner.

Typical usage
-------------
>>> from ABSim import simulate
>>> import pandas as pd

# Summary metrics only (default, fast)
>>> results = simulate(n_steps=2000, n_reps=10, burn_in=300)
>>> df = pd.DataFrame(results["summary"])
>>> df.groupby("rep")["boid_polarisation"].mean()

# Full trajectories for one repeat
>>> results = simulate(n_steps=500, n_reps=1, save_trajectories=True)
>>> traj = pd.DataFrame(results["trajectories"])

# Load a saved GUI session (speed, FOV, agent params, etc.) and add overrides
>>> results = simulate(n_reps=20, desired_speed=5.0,
...                    settings="~/my_experiment.json")
"""

import math
import json
import random
import time
from pathlib import Path

import numpy as np

import ABSim.config as config
import ABSim.metrics as metrics
from ABSim.agents import BOID
from ABSim.config import SPEED_LIMIT, CATCH_DIST, FOV
from ABSim.environment import Environment
from ABSim.model import Model

_SETTINGS_PATH = Path.home() / ".absim" / "settings.json"


def _load_file(path):
    try:
        with open(Path(path).expanduser()) as f:
            return json.load(f)
    except Exception:
        return {}


def simulate(
    n_steps=1000,
    n_reps=1,
    burn_in=200,
    agent_types=None,
    width=None,
    height=None,
    bounded=True,
    mask=None,
    desired_speed=None,
    fov_deg=None,
    catch_dist=None,
    save_trajectories=False,
    settings=None,
    seed=None,
    verbose=True,
):
    """
    Run one or more headless simulation repeats and return collected data.

    Parameters
    ----------
    n_steps           : frames to *record* per repeat (after burn-in)
    n_reps            : number of independent repeats
    burn_in           : frames to discard at the start of each repeat while
                        agents settle from their random initial positions
    agent_types       : list of AgentType; None = single default BOID group
    width, height     : arena size in pixels (None = from settings or 1920×1080)
    bounded           : canvas edges act as walls
    mask              : path to a wall-mask image (dark = wall, white = free)
    desired_speed     : agent speed (None = from settings / config default)
    fov_deg           : field of view in degrees (None = from settings)
    catch_dist        : predation catch distance (None = from settings)
    save_trajectories : also record per-agent x, y, heading every step
    settings          : str/Path → load this JSON file (merged on top of
                        ~/.absim/settings.json); or dict → use directly
    seed              : integer random seed for reproducibility
                        (each repeat gets seed+rep so they are independent
                        but the whole run is deterministic)
    verbose           : print one progress line per repeat

    Returns
    -------
    dict with keys
      "summary"      — list of dicts, one per (rep, step); convert with
                       pd.DataFrame(result["summary"])
                       Columns: rep, step, <group>_n, <group>_polarisation,
                                <group>_nnd, overall_polarisation
      "trajectories" — list of dicts (only when save_trajectories=True)
                       Columns: rep, step, id, group, x, y, heading_deg
    """
    # ── resolve settings ────────────────────────────────────────────────
    saved = _load_file(_SETTINGS_PATH)
    if isinstance(settings, (str, Path)):
        saved.update(_load_file(settings))
    elif isinstance(settings, dict):
        saved.update(settings)

    if desired_speed is None:
        desired_speed = saved.get("speed", SPEED_LIMIT)
    if fov_deg is None:
        fov_deg = saved.get("fov", FOV)
    if catch_dist is None:
        catch_dist = saved.get("catch_dist", CATCH_DIST)
    if width is None:
        width  = saved.get("width",  1920)
    if height is None:
        height = saved.get("height", 1080)
    bounded = saved.get("bounded", bounded)

    if agent_types is None:
        agent_types = [BOID]

    # ── output containers ───────────────────────────────────────────────
    summary_rows = []
    traj_rows    = [] if save_trajectories else None

    t_wall = time.perf_counter()

    for rep in range(n_reps):
        # Seed each repeat independently (reproducible but not correlated)
        if seed is not None:
            np.random.seed(seed + rep)
            random.seed(seed + rep)

        t_rep = time.perf_counter()

        config.WIDTH  = width
        config.HEIGHT = height

        env = Environment(
            width, height,
            bounded=bounded,
            avoid_radius=saved.get("wall_avoid_radius", 50.0),
            avoid_strength=saved.get("wall_strength",   6.0),
            thigmotaxis=saved.get("thigmotaxis", 0.0),
        )
        if mask is not None:
            env.load_mask(mask)

        model = Model(agent_types=agent_types)
        model.env = env
        for t in agent_types:
            model.init_agents(t.name, t.n_default)

        # ── burn-in: run without recording ──────────────────────────────
        for _ in range(burn_in):
            model.step(catch_dist=catch_dist,
                       desired_speed=desired_speed,
                       fov_deg=fov_deg)

        # ── recording loop ───────────────────────────────────────────────
        for step in range(n_steps):
            model.step(catch_dist=catch_dist,
                       desired_speed=desired_speed,
                       fov_deg=fov_deg)

            row = {"rep": rep, "step": step}
            row.update(metrics.compute(model.groups))
            summary_rows.append(row)

            if save_trajectories:
                for group_name, agents in model.groups.items():
                    for i, a in enumerate(agents):
                        traj_rows.append({
                            "rep":         rep,
                            "step":        step,
                            "id":          i,
                            "group":       group_name,
                            "x":           a.x,
                            "y":           a.y,
                            "heading_deg": math.degrees(math.atan2(a.dy, a.dx)) % 360,
                            **a.individual,   # spreads speed_scale, r_a, etc. as columns
                        })

        elapsed = time.perf_counter() - t_rep
        if verbose:
            counts = "  ".join(
                f"{name}={len(agents)}"
                for name, agents in model.groups.items()
            )
            print(f"  rep {rep + 1:>{len(str(n_reps))}}/{n_reps}"
                  f"  {elapsed:5.1f}s  {counts}")

    if verbose:
        total = time.perf_counter() - t_wall
        total_steps = n_reps * n_steps
        print(f"Done — {n_reps} rep{'s' if n_reps != 1 else ''} × "
              f"{n_steps:,} steps ({total_steps:,} total)  "
              f"{total:.1f}s  ({total_steps / total:,.0f} steps/s)")

    result = {"summary": summary_rows}
    if save_trajectories:
        result["trajectories"] = traj_rows
    return result
