"""Shared scenario configuration, independent of Tk and simulation execution."""
import copy
import json
from pathlib import Path
import warnings

_STATE_DIRECTORY = Path.home() / ".chimesim"
_ENVIRONMENT = {"width", "height", "bounded", "mask", "wall_shapes", "spawn_radius",
                "wall_avoid_radius", "wall_strength", "thigmotaxis"}
_MOVEMENT = {"desired_speed", "fov_deg", "repulsion_fov_deg", "visual_range", "capture_distance"}
_RUN = {"n_steps", "n_reps", "burn_in", "seed", "save_trajectories"}


def load_file(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Settings must contain a JSON object")
    scenario = data.get("scenario", data)
    if not isinstance(scenario, dict):
        raise ValueError("scenario must contain a JSON object")
    environment = scenario.get("environment", scenario)
    if not isinstance(environment, dict):
        raise ValueError("environment must contain a JSON object")
    if environment.get("mask"):
        mask = Path(environment["mask"]).expanduser()
        environment["mask"] = str(mask if mask.is_absolute() else path.parent / mask)
    return data


def flatten(data):
    """Translate the public scenario schema and legacy flat files internally."""
    data = load_file(data) if isinstance(data, (str, Path)) else copy.deepcopy(data)
    if not isinstance(data, dict):
        raise TypeError("settings must be a dictionary or JSON file path")
    if "scenario" in data:
        unknown = set(data) - {"scenario", "run", "conditions"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        result = flatten(data["scenario"])
        run = data.get("run", {})
        if not isinstance(run, dict) or set(run) - _RUN:
            raise ValueError("run contains unknown simulation controls")
        result.update(run)
        return result
    result = dict(data)
    for section, allowed in (("environment", _ENVIRONMENT), ("movement", _MOVEMENT)):
        if section in result:
            values = result.pop(section)
            if not isinstance(values, dict) or set(values) - (allowed | ({"catch_dist"} if section == "movement" else set())):
                raise ValueError(f"Unknown {section} settings")
            result.update(values)
    if 'catch_dist' in result:
        result.setdefault('capture_distance', result.pop('catch_dist'))
    if "model" in result:
        result["interaction_model"] = result.pop("model")
    return result


def scenario_from_flat(data):
    """Keep scientific scenario fields, excluding run and display preferences."""
    agents = copy.deepcopy(data["agents"])
    models = {agent.get("interaction_model", "zonal") for agent in agents.values()}
    model = data.get("interaction_model", next(iter(models)) if len(models) == 1 else None)
    return dict(model=model, environment={k: copy.deepcopy(data[k]) for k in _ENVIRONMENT if k in data},
                movement={k: data[k] for k in _MOVEMENT if k in data}, agents=agents)


def last_scenario():
    """Read the last interactive scenario, migrating the older combined file."""
    candidates = [_STATE_DIRECTORY / "scenario.json", _STATE_DIRECTORY / "settings.json",
                  Path.home() / ".absim" / "settings.json"]
    for path in candidates:
        if path.exists():
            try:
                from chimesim.simulate import _resolve_settings
                resolved, _ = _resolve_settings(settings=flatten(load_file(path)))
                return scenario_from_flat(resolved)
            except (OSError, ValueError, TypeError) as exc:
                warnings.warn(f"Could not restore scenario from {path}: {exc}", RuntimeWarning)
    return {}


def display_preferences():
    """Restore presentation independently of whichever scenario was selected."""
    scientific = _ENVIRONMENT | _MOVEMENT | _RUN | {"agents", "interaction_model"}
    candidates = [_STATE_DIRECTORY / "display.json", _STATE_DIRECTORY / "settings.json",
                  Path.home() / ".absim" / "settings.json"]
    for path in candidates:
        if path.exists():
            try:
                return {k: v for k, v in load_file(path).items() if k not in scientific}
            except (OSError, ValueError) as exc:
                warnings.warn(f"Could not restore display preferences: {exc}", RuntimeWarning)
    return {}


def save_session(data):
    """Persist scenario and display preferences in separate files."""
    from chimesim.simulate import _resolve_settings
    resolved, _ = _resolve_settings(settings=data)
    scenario = scenario_from_flat(resolved)
    scientific = _ENVIRONMENT | _MOVEMENT | _RUN | {"agents", "interaction_model"}
    preferences = {k: v for k, v in data.items() if k not in scientific}
    _STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, values in (("scenario.json", scenario), ("display.json", preferences)):
        target = _STATE_DIRECTORY / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)


def settings(source="last", *, model=None, environment=None, movement=None, agents=None):
    """Build a validated, independent scenario dictionary.

    source="last" restores the last interactive scenario, or package defaults.
    source="defaults" starts fresh. A dictionary or JSON path starts from that
    configuration, without inheriting the last session. Supplied agent types
    replace the population; environment and movement fields override the source.
    """
    from chimesim.simulate import _resolve_settings
    if isinstance(source, str) and source in {"last", "defaults"}:
        base = last_scenario() if source == "last" else {}
    else:
        base = source
    flat = flatten(base)
    # Match the interactive starting range for independent scripted scenarios.
    flat.setdefault("visual_range", 200.0)
    if model is not None:
        flat["interaction_model"] = model
    for section, values, allowed in (("environment", environment, _ENVIRONMENT),
                                      ("movement", movement, _MOVEMENT)):
        if values is not None:
            if not isinstance(values, dict) or set(values) - (allowed | ({"catch_dist"} if section == "movement" else set())):
                raise ValueError(f"Unknown {section} settings")
            values = copy.deepcopy(values)
            if section == 'movement' and 'catch_dist' in values:
                values.setdefault('capture_distance', values.pop('catch_dist'))
            flat.update(values)
    if agents is not None:
        flat["agents"] = copy.deepcopy(agents)
    elif "agents" not in flat:
        from chimesim.agents import BOID
        flat["agents"] = {"A1": BOID.clone("A1").to_dict()}
    resolved, _ = _resolve_settings(settings=flat)
    return scenario_from_flat(resolved)
