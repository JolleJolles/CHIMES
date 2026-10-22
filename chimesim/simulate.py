"""Reproducible headless simulations from keyword arguments, dictionaries or JSON."""

import math
import time
from pathlib import Path

from chimesim.agents import AgentType, BOID
from chimesim.environment import Environment
from chimesim.model import Model
from chimesim import metrics
from chimesim.configuration import load_file as _load_file, flatten


def _resolve_settings(n_steps=None, n_reps=None, burn_in=None, agent_types=None,
             width=None, height=None, bounded=None, mask=None,
             desired_speed=None, fov_deg=None, catch_dist=None,
             save_trajectories=None, settings=None, seed=None, verbose=True,
             wall_avoid_radius=None, wall_strength=None, thigmotaxis=None, visual_range=None, repulsion_fov_deg=None, spawn_radius=None):
    """Resolve configuration without running any simulation steps."""
    if settings is None:
        saved = {}
    elif isinstance(settings, (str, Path)):
        saved = flatten(settings)
    elif isinstance(settings, dict):
        saved = flatten(settings)
    else:
        raise TypeError("settings must be a dictionary or a JSON file path")

    # Accept saved GUI names as aliases for Python arguments.
    for old, new in (("speed", "desired_speed"), ("fov", "fov_deg")):
        if old in saved:
            saved.setdefault(new, saved.pop(old))
    defaults = dict(n_steps=1000, n_reps=1, burn_in=200, width=1920, height=1080,
                    bounded=True, mask=None, desired_speed=3.0, fov_deg=270.0,
                    catch_dist=40.0, save_trajectories=False, seed=None,
                    wall_avoid_radius=50.0, wall_strength=6.0, thigmotaxis=0.0, visual_range=None, repulsion_fov_deg=None, spawn_radius=None)
    arguments = locals().copy()
    unknown = set(saved) - set(defaults) - {"zone_panel", "show_scale_bar", "show_visual_neighbours", "visual_range", "visual_fraction", "display_size", "body_style", "wall_shapes", "agents", "type_count", "show_edges", "show_trails", "trail_length", "group_distance_multiplier", "position_colour", "position_order_by", "position_view", "window_width", "window_height", "window_layout", "interaction_model", "lock_arena_size", "record_trajectories"}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    resolved = {key: arguments[key] if arguments[key] is not None else saved.get(key, value)
                for key, value in defaults.items()}
    for key in ("n_steps", "n_reps", "burn_in", "width", "height"):
        value = resolved[key]
        minimum = 0 if key == "burn_in" else 2 if key in {"width", "height"} else 1
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
    if resolved["visual_range"] is not None and (not math.isfinite(resolved["visual_range"]) or resolved["visual_range"] < 0):
        raise ValueError("visual_range must be finite and nonnegative or None")
    if resolved['repulsion_fov_deg'] is not None and (not math.isfinite(resolved['repulsion_fov_deg']) or not 0 <= resolved['repulsion_fov_deg'] <= 360):
        raise ValueError('repulsion_fov_deg must be between 0 and 360 or None')
    if resolved['spawn_radius'] is not None and (not math.isfinite(resolved['spawn_radius']) or resolved['spawn_radius'] <= 0):
        raise ValueError('spawn_radius must be finite and positive or None')
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
    if saved.get("interaction_model") is not None:
        if saved["interaction_model"] not in {"zonal", "voronoi"}:
            raise ValueError("interaction_model must be 'zonal' or 'voronoi'")
        for t in agent_types:
            t.interaction_model = saved["interaction_model"]
    resolved["wall_shapes"] = saved.get("wall_shapes", [])
    resolved["agents"] = {t.name: t.to_dict() for t in agent_types}
    if resolved["mask"] is not None:
        resolved["mask"] = str(Path(resolved["mask"]).expanduser().resolve())

    for agent_type in agent_types:
        agent_type.validate()
    return resolved, agent_types


def _simulate_single(n_steps=None, n_reps=None, burn_in=None, agent_types=None,
             width=None, height=None, bounded=None, mask=None,
             desired_speed=None, fov_deg=None, catch_dist=None,
             save_trajectories=None, settings=None, seed=None, verbose=True,
             wall_avoid_radius=None, wall_strength=None, thigmotaxis=None, visual_range=None, repulsion_fov_deg=None, spawn_radius=None,
             progress=None, should_cancel=None, on_row=None, collect_results=True, performance=False):
    """Shared step engine for one resolved condition, with streaming/cancellation."""
    configuration = locals().copy()
    for key in ('progress', 'should_cancel', 'on_row', 'collect_results', 'performance'):
        configuration.pop(key)
    resolved, agent_types = _resolve_settings(**configuration)
    seed = resolved['seed']
    summary_rows, traj_rows, individual_rows, repeat_rows = [], [], [], []
    started = time.perf_counter()
    cpu_started = time.process_time() if performance else None
    repeat_reports = []
    agent_updates = 0
    completed = 0
    completed_reps = 0
    cancelled = False
    last_progress = -float('inf')
    total = resolved['n_reps'] * (resolved['burn_in'] + resolved['n_steps'])
    def report(rep, phase, step, force=False):
        nonlocal last_progress
        now = time.perf_counter()
        if progress is not None and (force or now - last_progress >= .1):
            progress(dict(completed=completed, total=total, repeat=rep + 1,
                          repeats=resolved['n_reps'], phase=phase, step=step,
                          elapsed=now - started))
            last_progress = now
    def stopping():
        return should_cancel is not None and should_cancel()
    report(0, 'Preparing', 0, True)
    for rep in range(resolved["n_reps"]):
        if stopping():
            cancelled = True
            break
        if performance:
            repeat_started, repeat_cpu = time.perf_counter(), time.process_time()
            previous_steps, previous_updates = completed, agent_updates
        env = Environment(resolved["width"], resolved["height"], bounded=resolved["bounded"],
                          mask=resolved["mask"], avoid_radius=resolved["wall_avoid_radius"],
                          avoid_strength=resolved["wall_strength"], thigmotaxis=resolved["thigmotaxis"], wall_shapes=resolved["wall_shapes"], spawn_radius=resolved["spawn_radius"])
        model = Model(agent_types=[t.clone() for t in agent_types],
                      seed=None if seed is None else seed + rep)
        model.env = env
        for t in model.agent_types.values():
            model.init_agents(t.name, t.n_default)
        set_id = f"c0_r{rep}"
        # Static assignments are written once, including agents lost during burn-in.
        for name, agents in model.groups.items():
            traits = {key: value for key, value in model.agent_types[name].to_dict().items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool) and key != 'n_default'}
            for agent in agents:
                individual = dict(set=set_id, condition=0, rep=rep, id=agent.unique_id, agent_type=name,
                                  **{key: agent.individual.get(key, value) for key, value in traits.items()})
                if collect_results:
                    individual_rows.append(individual)
                if on_row is not None:
                    on_row('individuals', individual)
        metric_totals = {key: 0. for key in metrics.compute({name: [] for name in model.groups}, env)}
        recorded_steps = 0
        step_args = dict(catch_dist=resolved["catch_dist"], desired_speed=resolved["desired_speed"],
                         fov_deg=resolved["fov_deg"], visual_range=resolved["visual_range"], repulsion_fov_deg=resolved["repulsion_fov_deg"])
        for step in range(resolved["burn_in"]):
            if stopping():
                cancelled = True
                break
            if performance:
                agent_updates += sum(len(agents) for agents in model.groups.values())
            model.step(**step_args)
            completed += 1
            report(rep, 'Burn-in', step + 1)
        if not cancelled:
            report(rep, 'Recording', 0, True)
        for step in range(0 if cancelled else resolved["n_steps"]):
            if stopping():
                cancelled = True
                break
            if performance:
                agent_updates += sum(len(agents) for agents in model.groups.values())
            model.step(**step_args)
            completed += 1
            values = metrics.compute(model.groups, env)
            recorded_steps += 1
            for key, value in values.items():
                metric_totals[key] = metric_totals.get(key, 0.) + value
            frame = resolved["burn_in"] + step + 1
            row = dict(set=set_id, condition=0, rep=rep, step=step, frame=frame, t=frame, **values)
            if collect_results:
                summary_rows.append(row)
            if on_row is not None:
                on_row('summary', row)
            if resolved["save_trajectories"]:
                for name, agents in model.groups.items():
                    for agent in agents:
                        frame = resolved["burn_in"] + step + 1
                        row = dict(set=set_id, condition=0, rep=rep, frame=frame, t=frame,
                                   id=agent.unique_id, x=agent.x, y=agent.y,
                                   vx=agent.dx, vy=agent.dy, speed=math.hypot(agent.dx, agent.dy),
                                   heading_rad=agent.heading % (2 * math.pi))
                        if collect_results:
                            traj_rows.append(row)
                        if on_row is not None:
                            on_row('trajectories', row)
            report(rep, 'Recording', step + 1)
        repeat_row = dict(set=set_id, condition=0, rep=rep, seed=seed + rep if seed is not None else None,
                          recorded_steps=recorded_steps, cancelled=cancelled,
                          **{key + '_mean': value / recorded_steps if recorded_steps else None for key, value in metric_totals.items()})
        if collect_results:
            repeat_rows.append(repeat_row)
        if on_row is not None:
            on_row('repeats', repeat_row)
        if performance:
            from chimesim.performance import workload
            repeat_reports.append(workload(time.perf_counter()-repeat_started, time.process_time()-repeat_cpu,
                                           completed-previous_steps, agent_updates-previous_updates,
                                           rep=rep, seed=seed+rep, cancelled=cancelled))
        if cancelled:
            break
        completed_reps += 1
        report(rep, 'Repeat complete', resolved['n_steps'], True)
        if verbose:
            print(f"Completed repeat {rep + 1}/{resolved['n_reps']}")
    if verbose:
        print(f"Done in {time.perf_counter() - started:.1f}s")
    report(min(completed_reps, resolved['n_reps'] - 1), 'Cancelled' if cancelled else 'Complete', 0, True)
    result = {"summary": summary_rows, "individuals": individual_rows, "repeats": repeat_rows, "settings": resolved}
    if progress is not None or should_cancel is not None or on_row is not None:
        result.update(cancelled=cancelled, completed_reps=completed_reps, completed_steps=completed)
    if resolved["save_trajectories"]:
        result["trajectories"] = traj_rows
    if performance:
        from chimesim.performance import workload
        result['_performance'] = workload(time.perf_counter()-started, time.process_time()-cpu_started,
                                          completed, agent_updates, completed_reps=completed_reps,
                                          cancelled=cancelled, repeats=repeat_reports,
                                          initial_agents=sum(t.n_default for t in agent_types))
    return result


def simulate(settings=None, *, n_steps=None, n_reps=None, burn_in=None, seed=None,
             save_trajectories=None, conditions=None, gui=False, output_dir=None, output=None, performance=False, **overrides):
    """Generate datasets, open setup with gui=True, or run a condition grid.

    gui=True returns the latest saved-run information after Exit, or None if no
    simulation ran. The return value or its output_dir can be loaded or plotted.
    With no explicit scenario, reuse the last interactive scenario. Explicit
    settings or scenario overrides start independently of that session. Repeats
    use consecutive seeds; conditions form a Cartesian grid of parameter values.
    output_dir selects a parent directory; each call creates a unique run folder
    containing CSVs/configuration/status instead of holding
    datasets in memory. performance=True adds per-condition/repeat timings,
    workload rates and computer details; output mode also saves performance.json.
    burn_in (default 200) precedes n_steps recorded updates. Trajectories contain
    dynamic state; individuals store assigned traits once, and repeats store
    time-averaged metrics. The older output keyword aliases output_dir.
    """
    if output is not None:
        if output_dir is not None:
            raise ValueError("Use output_dir or its older output alias, not both")
        output_dir = output
    if not isinstance(performance, bool):
        raise TypeError("performance must be True or False")
    import secrets
    from chimesim.configuration import settings as make_settings, scenario_from_flat, _RUN
    runtime_names = {"progress", "should_cancel", "on_row", "collect_results", "verbose"}
    runtime = {k: overrides.pop(k) for k in list(overrides) if k in runtime_names}
    if settings is None:
        settings = make_settings(source="defaults" if overrides else "last")
    if conditions is None and isinstance(settings, (dict, str, Path)):
        supplied = _load_file(settings) if isinstance(settings, (str, Path)) else settings
        conditions = supplied.get("conditions")
    arguments = dict(settings=settings, n_steps=n_steps, n_reps=n_reps, burn_in=burn_in,
                     seed=seed, save_trajectories=save_trajectories, **overrides)
    resolved, _ = _resolve_settings(**arguments)
    if not isinstance(gui, bool):
        raise TypeError("gui must be True or False")
    # Validate all grid combinations before creating files or opening a GUI.
    scenarios = _condition_grid(scenario_from_flat(resolved), conditions)
    if gui:
        if any(k != "verbose" for k in runtime) or output_dir is not None:
            raise ValueError("Use callback/streaming and output options in direct simulate() mode; the GUI manages its own worker")
        from chimesim.experiment import launch
        return launch(settings=resolved, conditions=conditions, **({"performance": True} if performance else {}))
    if resolved["seed"] is None:
        resolved["seed"] = secrets.randbits(32)
    if output_dir is not None:
        if "on_row" in runtime or "collect_results" in runtime:
            raise ValueError("output manages streaming and collection itself")
        from chimesim.experiment import run_experiment
        return run_experiment(resolved, output_dir, conditions=conditions,
                              progress=runtime.get("progress"), should_cancel=runtime.get("should_cancel"), performance=performance)
    performance_started = time.perf_counter() if performance else None
    performance_cpu = time.process_time() if performance else None
    run = {k: resolved[k] for k in _RUN}
    if not conditions:
        result = _simulate_single(settings=resolved, performance=performance, **runtime)
        result["settings"] = scenario_from_flat(resolved)
        result["run"] = run
        if performance:
            from chimesim.performance import report
            entry = dict(condition=0, values={}, **result.pop('_performance'))
            result['performance'] = report([entry], time.perf_counter()-performance_started,
                                           time.process_time()-performance_cpu, verbose=runtime.get('verbose', True))
        return result
    started = time.perf_counter()
    total_per = run["n_reps"] * (run["burn_in"] + run["n_steps"])
    summary, trajectories, individuals, repeats = [], [], [], []
    configurations = [dict(condition=i, values=values, scenario=scenario,
                           run=dict(run, seed=run["seed"] + i * run["n_reps"]))
                      for i, (scenario, values) in enumerate(scenarios)]
    condition_reports = []
    completed_steps = completed_reps = 0
    cancelled = False
    progress = runtime.pop("progress", None)
    on_row = runtime.pop("on_row", None)
    for index, (scenario, values) in enumerate(scenarios):
        condition_run = dict(run, seed=run["seed"] + index * run["n_reps"])
        config = dict(scenario=scenario, run=condition_run)
        def report(update):
            if progress:
                progress({**update, "condition": index, "conditions": len(scenarios),
                          "completed": completed_steps + update["completed"],
                          "total": len(scenarios) * total_per, "elapsed": time.perf_counter()-started})
        def row(kind, value):
            if on_row:
                on_row(kind, indexed(value))
        def indexed(value):
            value = dict(value, condition=index)
            if 'set' in value:
                value['set'] = f"c{index}_r{value['rep']}"
            return value
        result = _simulate_single(settings=config, progress=report, on_row=row, performance=performance, **runtime)
        if performance:
            condition_reports.append(dict(condition=index, values=values, **result.pop("_performance")))
        summary.extend(indexed(v) for v in result["summary"])
        trajectories.extend(indexed(v) for v in result.get("trajectories", []))
        individuals.extend(indexed(v) for v in result["individuals"])
        repeats.extend(indexed(v) for v in result["repeats"])
        completed_steps += result["completed_steps"]
        completed_reps += result["completed_reps"]
        cancelled = result["cancelled"]
        if cancelled:
            break
    result = dict(summary=summary, individuals=individuals, repeats=repeats, settings=scenario_from_flat(resolved), run=run,
                  conditions=configurations, cancelled=cancelled,
                  completed_steps=completed_steps, completed_reps=completed_reps)
    if run["save_trajectories"]:
        result["trajectories"] = trajectories
    if performance:
        from chimesim.performance import report as performance_report
        result['performance'] = performance_report(condition_reports, time.perf_counter()-performance_started,
                                                    time.process_time()-performance_cpu, verbose=runtime.get('verbose', True))
    return result


def _condition_grid(scenario, conditions):
    """Resolve and validate every Cartesian-grid combination without mutations."""
    import copy
    import itertools
    from chimesim.configuration import settings as make_settings
    if conditions is None or conditions == {}:
        return [(scenario, {})]
    if not isinstance(conditions, dict):
        raise TypeError("conditions must map dotted parameter paths to lists of values")
    for path, values in conditions.items():
        if not isinstance(path, str) or not isinstance(values, (list, tuple)) or not values:
            raise ValueError("Each condition needs a dotted parameter path and a nonempty list")
    result = []
    for combination in itertools.product(*conditions.values()):
        candidate = copy.deepcopy(scenario)
        values = dict(zip(conditions, combination))
        for path, value in values.items():
            parts = path.split(".")
            if len(parts) == 3 and parts[0] == "agents":
                parts[-1] = {"n": "n_default", "repulsion": "r_r", "orientation": "r_o", "attraction": "r_a"}.get(parts[-1], parts[-1])
            target = candidate
            for part in parts[:-1]:
                if not isinstance(target, dict) or part not in target:
                    raise ValueError(f"Unknown condition parameter: {path}")
                target = target[part]
            if not isinstance(target, dict) or parts[-1] not in target:
                raise ValueError(f"Unknown condition parameter: {path}")
            target[parts[-1]] = copy.deepcopy(value)
        result.append((make_settings(source=candidate), values))
    return result
