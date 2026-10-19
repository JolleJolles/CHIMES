"""Reproducible headless simulations from keyword arguments, dictionaries or JSON."""

import json
import math
import time
from pathlib import Path

from ABSim.agents import AgentType, BOID
from ABSim.environment import Environment
from ABSim.model import Model
from ABSim import metrics


def _load_file(path):
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise ValueError("Settings must contain a JSON object")
    if settings.get("mask"):
        mask = Path(settings["mask"]).expanduser()
        settings["mask"] = str(mask if mask.is_absolute() else path.parent / mask)
    return settings


def simulate(n_steps=None, n_reps=None, burn_in=None, agent_types=None,
             width=None, height=None, bounded=None, mask=None,
             desired_speed=None, fov_deg=None, catch_dist=None,
             save_trajectories=None, settings=None, seed=None, verbose=True,
             wall_avoid_radius=None, wall_strength=None, thigmotaxis=None):
    """Run repeats and return summary rows, resolved settings and optional trajectories.

    Explicit arguments override settings (a dict or JSON path), then defaults.
    GUI settings can be loaded explicitly with settings="~/.absim/settings.json".
    Use an ``agents`` mapping of names to AgentType constructor parameters in
    settings, or pass a list of AgentType objects as ``agent_types``.
    Angles in Python/configuration use radians except fov_deg; legacy GUI
    max_turn_deg/noise_deg values are also accepted. Relative mask paths in
    JSON are resolved relative to that file.

    n_steps is the number of recorded steps AFTER burn_in. Each repeat uses
    seed + rep. Returned settings can be saved as JSON and passed back into
    simulate(). Trajectory IDs are unique within each repeat; use (rep, id).
    """
    import copy
    if settings is None:
        saved = {}
    elif isinstance(settings, (str, Path)):
        saved = _load_file(settings)
    elif isinstance(settings, dict):
        saved = copy.deepcopy(settings)
    else:
        raise TypeError("settings must be a dictionary or a JSON file path")

    # Accept saved GUI names as aliases for Python arguments.
    for old, new in (("speed", "desired_speed"), ("fov", "fov_deg")):
        if old in saved:
            saved.setdefault(new, saved.pop(old))
    defaults = dict(n_steps=1000, n_reps=1, burn_in=200, width=1920, height=1080,
                    bounded=True, mask=None, desired_speed=3.0, fov_deg=270.0,
                    catch_dist=40.0, save_trajectories=False, seed=None,
                    wall_avoid_radius=50.0, wall_strength=6.0, thigmotaxis=0.0)
    arguments = locals().copy()
    unknown = set(saved) - set(defaults) - {"agents", "type_count", "show_edges"}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    resolved = {key: arguments[key] if arguments[key] is not None else saved.get(key, value)
                for key, value in defaults.items()}
    for key in ("n_steps", "n_reps", "burn_in", "width", "height"):
        value = resolved[key]
        minimum = 0 if key == "burn_in" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("desired_speed", "catch_dist", "wall_avoid_radius", "wall_strength", "thigmotaxis"):
        if not math.isfinite(resolved[key]) or resolved[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(resolved["fov_deg"]) or not 0 <= resolved["fov_deg"] <= 360:
        raise ValueError("fov_deg must be between 0 and 360 degrees")
    for key in ("bounded", "save_trajectories"):
        if not isinstance(resolved[key], bool):
            raise ValueError(f"{key} must be a boolean")
    seed = resolved["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if agent_types is None:
        if "agents" in saved:
            agent_types = [AgentType.from_dict(name, values) for name, values in saved["agents"].items()]
        else:
            agent_types = [BOID.clone()]
    else:
        agent_types = [agent.clone() for agent in agent_types]
    if not agent_types or len({t.name for t in agent_types}) != len(agent_types):
        raise ValueError("Provide at least one agent type with unique names")
    resolved["agents"] = {t.name: t.to_dict() for t in agent_types}
    if resolved["mask"] is not None:
        resolved["mask"] = str(Path(resolved["mask"]).expanduser().resolve())

    summary_rows, traj_rows = [], []
    started = time.perf_counter()
    for rep in range(resolved["n_reps"]):
        env = Environment(resolved["width"], resolved["height"], bounded=resolved["bounded"],
                          mask=resolved["mask"], avoid_radius=resolved["wall_avoid_radius"],
                          avoid_strength=resolved["wall_strength"], thigmotaxis=resolved["thigmotaxis"])
        model = Model(agent_types=[t.clone() for t in agent_types],
                      seed=None if seed is None else seed + rep)
        model.env = env
        for t in model.agent_types.values():
            model.init_agents(t.name, t.n_default)
        step_args = dict(catch_dist=resolved["catch_dist"], desired_speed=resolved["desired_speed"],
                         fov_deg=resolved["fov_deg"])
        for _ in range(resolved["burn_in"]):
            model.step(**step_args)
        for step in range(resolved["n_steps"]):
            model.step(**step_args)
            summary_rows.append({"rep": rep, "step": step, **metrics.compute(model.groups, env)})
            if resolved["save_trajectories"]:
                for name, agents in model.groups.items():
                    for agent in agents:
                        traj_rows.append({"rep": rep, "step": step, "id": agent.unique_id,
                                          "group": name, "x": agent.x, "y": agent.y,
                                          "heading_deg": math.degrees(math.atan2(agent.dy, agent.dx)) % 360,
                                          **agent.individual})
        if verbose:
            print(f"Completed repeat {rep + 1}/{resolved['n_reps']}")
    if verbose:
        print(f"Done in {time.perf_counter() - started:.1f}s")
    result = {"summary": summary_rows, "settings": resolved}
    if resolved["save_trajectories"]:
        result["trajectories"] = traj_rows
    return result
