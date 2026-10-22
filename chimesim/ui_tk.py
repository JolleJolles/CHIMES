"""Interactive windows for viewing a fixed arena and controlling its model."""

import tkinter as tk
from tkinter import messagebox, filedialog
import math
import time
import json
import warnings
import re
import csv
import copy
import shutil
import tempfile
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError
from chimesim.model import Model
from chimesim.agents import AgentType, BOID
from chimesim.config import SPEED_LIMIT, CAPTURE_DISTANCE, FOV
from chimesim import metrics
from chimesim.recording import MouseTrajectory
from chimesim.ui_events import bind_wheel, wheel_units, choice, popup_active, show_window, destroy_window, arena_pointer
from chimesim.ui_zones import InteractionZones
from chimesim.ui_positions import PositionHistory
from chimesim.ui_visual_network import VisualNetwork, wrapped_segments
from chimesim.body_geometry import body_geometry, body_size
from chimesim.environment import Environment
from chimesim.environment_drawing import WallEditor, draw_walls
from chimesim.simulate import _load_file
from chimesim.configuration import flatten, scenario_from_flat, display_preferences

_MAX_TYPES = 3

_TRAIT_LABELS = {
    "speed_scale": "Speed multiplier", "r_r": "Repulsion range", "r_o": "Orientation range",
    "r_a": "Attraction / max range", "flee_dist": "Threat detection range", "social_speed": "Social speed",
    "preferred_distance": "Preferred spacing", "speed_relaxation": "Speed recovery",
    "speed_noise": "Speed noise", "max_speed_scale": "Max speed multiplier",
    "max_turn": "Max turn", "noise": "Heading noise",
}


def _arena_transform(width, height, view_width, view_height):
    """Uniform scale and centred offsets; viewing never changes model geometry."""
    scale = min(max(1, view_width) / width, max(1, view_height) / height)
    return scale, (view_width - width * scale) / 2, (view_height - height * scale) / 2


def _load_settings():
    from chimesim.configuration import last_scenario
    return {**display_preferences(), **flatten(last_scenario())}


def _save_settings(data):
    from chimesim.configuration import save_session
    try:
        save_session(data)
    except (OSError, ValueError) as exc:
        warnings.warn(f"Could not save session: {exc}", RuntimeWarning)


def _prepare_model(saved, agent_types=None, width=None, height=None, bounded=None, mask=None, seed=None):
    if agent_types is not None:
        all_types = [t.clone() for t in agent_types]
    elif saved.get("agents"):
        all_types = [AgentType.from_dict(name, values) for name, values in saved["agents"].items()]
    else:
        all_types = [BOID.clone()]
    if not 1 <= len(all_types) <= _MAX_TYPES or len({t.name for t in all_types}) != len(all_types):
        raise ValueError(f"The GUI supports 1–{_MAX_TYPES} uniquely named agent types")
    names = {t.name: f"A{i}" for i, t in enumerate(all_types, 1)}
    if all(t.name in {"A1", "A2", "A3"} for t in all_types):
        names = {t.name: t.name for t in all_types}
    for agent_type in all_types:
        agent_type.name = names[agent_type.name]
        agent_type.pursues = {names.get(name, name) for name in agent_type.pursues}
        agent_type.flees = {names.get(name, name) for name in agent_type.flees}
        agent_type.captures = {names.get(name, name) for name in agent_type.captures}
        agent_type.validate()
    width = saved.get("width", 1920) if width is None else width
    height = saved.get("height", 1080) if height is None else height
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (width, height)):
        raise ValueError("Arena width and height must be integers >= 2")
    bounded = saved.get("bounded", True) if bounded is None else bounded
    mask = saved.get("mask") if mask is None else mask
    seed = saved.get("seed") if seed is None else seed
    multiplier = saved.get("group_distance_multiplier", 2.5)
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("group_distance_multiplier must be finite and positive")
    model = Model(agent_types=all_types, seed=seed)
    model.env = Environment(width, height, bounded=bounded, mask=mask,
                            avoid_radius=saved.get("wall_avoid_radius", 50.0),
                            avoid_strength=saved.get("wall_strength", 6.0),
                            thigmotaxis=saved.get("thigmotaxis", 0.0), wall_shapes=saved.get("wall_shapes", []), spawn_radius=saved.get("spawn_radius"))
    visual_range = (saved.get("visual_range") if saved.get("visual_range") is not None else 200.)
    if not math.isfinite(visual_range) or visual_range < 0:
        raise ValueError("visual_range must be finite and nonnegative")
    for t in all_types:
        t.constrain_zones(visual_range * t.visual_range_multiplier)
        model.init_agents(t.name, t.n_default)

    return all_types, model, width, height, bounded, mask, seed


def interactive(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None,
        width=None, height=None, window_width=None, window_height=None):
    """Run the interactive GUI; settings accepts a dictionary or JSON path.

    Load settings opens a fresh paused session. With log=True, return the metrics
    from the final session. See _interactive_session for arena and window behaviour.
    """
    arguments = dict(agent_types=agent_types, log=log, mask=mask, bounded=bounded,
                     settings=settings, seed=seed, width=width, height=height,
                     window_width=window_width, window_height=window_height)
    while True:
        rows, loaded = _interactive_session(**arguments)
        if loaded is None:
            return rows
        arguments = dict(settings=loaded, log=log)


def _interactive_session(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None,
                 width=None, height=None, window_width=None, window_height=None):
    """Open simulation, controls, information and per-agent-type windows.

    width/height are initial arena dimensions in model pixels. They stay fixed
    when Lock arena size is selected. Otherwise resizing rescales the
    arena after movement settles. window_width/window_height set the initial
    visualisation size in screen pixels. Explicit arguments override dictionary/JSON settings. With
    settings=None, restore the last GUI session. Lock arena size fixes the space;
    Apply size & reset changes the arena and clears recorded data. Closing the
    simulation exits; closing a settings panel hides it; closing an agent panel removes its type (the last type is only hidden). log=True returns metrics
    for each model step after closing. Seed controls initial randomisation.
    """
    if settings is None:
        saved = _load_settings()
    elif isinstance(settings, (str, Path)):
        saved = {**display_preferences(), **flatten(_load_file(settings))}
    elif isinstance(settings, dict):
        saved = {**display_preferences(), **flatten(settings)}
    else:
        raise TypeError("settings must be a dictionary or JSON path")
    all_types, model, width, height, bounded, mask, seed = _prepare_model(
        saved, agent_types, width, height, bounded, mask, seed)
    source_names = list(saved.get("agents", {})) if agent_types is None else [t.name for t in agent_types]
    names = dict(zip(source_names, [t.name for t in all_types]))
    root = tk.Tk()
    root.title("Simulation")
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    view_w = saved.get("window_width", min(width, int(screen_w * .6))) if window_width is None else window_width
    view_h = saved.get("window_height", min(height, int(screen_h * .7))) if window_height is None else window_height
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (view_w, view_h)):
        root.destroy()
        raise ValueError("Window width and height must be integers >= 2")
    left_w, right_w = 340, 330
    cascade = 26
    control_h = min(650 + 70 * (len(all_types) - 1), max(300, int(screen_h * .52)))
    sim_x = left_w + 40
    default_view_w = max(200, screen_w - sim_x - right_w - 110)
    if window_width is None and not saved.get("window_layout"):
        view_w = min(view_w, default_view_w)
    right_x = min(sim_x + view_w + 20, max(sim_x, screen_w - right_w - 90))
    layout = dict(saved.get("window_layout", {}))
    for old_name, name in names.items():
        if old_name != name and f"agent:{old_name}" in layout:
            layout[f"agent:{name}"] = layout.pop(f"agent:{old_name}")
    windows = {}
    if "general_movement" not in layout:
        # Make room for the new left-hand stack on the first upgraded launch.
        layout.pop("controls", None)
        layout.pop("information", None)

    def _place(window, key, geometry):
        windows[key] = window
        stored = layout.get(key, {})
        candidate = stored.get("geometry", geometry)
        match = re.fullmatch(r"(\d+)x(\d+)[+-]\d+[+-]\d+", str(candidate))
        geometry = candidate if match and int(match[1])>=160 and int(match[2])>=120 else geometry
        window.geometry(geometry)
        layout[key] = dict(geometry=geometry, visible=stored.get('visible',True))
        if stored.get("visible", True) is False:
            window.withdraw()
        window.protocol("WM_DELETE_WINDOW", lambda: _hide(window, key))

    def _hide(window, key):
        if key == "controls":
            exit_sim()
            return
        layout[key] = dict(geometry=window.geometry(), visible=False)
        window.withdraw()
        if key == "information":
            show_info_var.set(False)
        if key == "positions":
            show_positions_var.set(False)
        _save_settings(_collect_settings())

    _place(root, "simulation", f"{view_w}x{view_h}+{sim_x}+40")
    root.minsize(200, 150)
    canvas = tk.Canvas(root, bg="#e8e8e8", highlightthickness=0)
    horizontal = tk.Scrollbar(root, orient=tk.HORIZONTAL)
    horizontal.pack(side=tk.BOTTOM, fill=tk.X)
    vertical = tk.Scrollbar(root, orient=tk.VERTICAL)
    vertical.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    controls = tk.Toplevel(root)
    controls.title("Controls")
    _place(controls, "controls", f"{left_w}x{control_h}+20+40")
    parameters = tk.Toplevel(root)
    parameters.title("General movement settings")
    # Migrate the old right-hand settings window once; subsequent moves are saved.
    layout.pop("parameters", None)
    parameter_y = 40 + control_h + 30
    _place(parameters, "general_movement", f"{left_w}x{max(280, screen_h - parameter_y - 70)}+20+{parameter_y}")
    info = tk.Toplevel(root)
    info.title("Information")
    info.minsize(340,280)
    _place(info, "information", f"440x{min(900, screen_h - 120)}+{sim_x}+80")
    info_scroll = tk.Scrollbar(info)
    info_canvas = tk.Canvas(info, bg="white", highlightthickness=0, yscrollcommand=info_scroll.set)
    info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    info_scroll.config(command=info_canvas.yview)
    def _info_scroll_visibility(first, last):
        info_scroll.set(first, last)
        if float(first) <= 0 and float(last) >= 1:
            info_scroll.pack_forget()
        elif not info_scroll.winfo_manager():
            info_scroll.pack(side=tk.RIGHT, fill=tk.Y, before=info_canvas)
    info_canvas.config(yscrollcommand=_info_scroll_visibility)
    bind_wheel(info, lambda event: info_canvas.yview_scroll(wheel_units(event), "units"))

    state = dict(active=False, moving=False, timer=None, settle=None, remaining=0,
                 step=0, fps=0., last_frame=time.perf_counter(), last_info=0., closing=False,
                 reload_settings=None, mouse_pointer=None)
    rows = [] if log else None
    status = tk.StringVar(value="Paused")
    n_vars, panels, panel_counts, zone_labels = {}, {}, {}, {}
    speed_var = tk.DoubleVar(value=saved.get("desired_speed", saved.get("speed", SPEED_LIMIT)))
    repulsion_fov_var = tk.DoubleVar(value=saved.get("repulsion_fov_deg") if saved.get("repulsion_fov_deg") is not None else saved.get("fov_deg", FOV))
    fov_var = tk.DoubleVar(value=saved.get("fov_deg", saved.get("fov", FOV)))
    capture_distance_var = tk.DoubleVar(value=saved.get("capture_distance", CAPTURE_DISTANCE))
    bounded_var = tk.BooleanVar(value=bounded)
    wall_ar_var = tk.DoubleVar(value=model.env.avoid_radius)
    wall_as_var = tk.DoubleVar(value=model.env.avoid_strength)
    wall_thig_var = tk.DoubleVar(value=model.env.thigmotaxis)
    mouse_predator_var = tk.BooleanVar(value=saved.get("mouse_predator", False))
    show_visual_var = tk.BooleanVar(value=saved.get("show_visual_neighbours", False))
    visual_range_var = tk.DoubleVar(value=(saved.get("visual_range") if saved.get("visual_range") is not None else 200.))
    visual_fraction_var = tk.DoubleVar(value=saved.get("visual_fraction", .05))
    show_edges_var = tk.BooleanVar(value=saved.get("show_edges", False))
    show_info_var = tk.BooleanVar(value=info.state() != "withdrawn")
    show_scale_var = tk.BooleanVar(value=saved.get("show_scale_bar", True))
    show_trails_var = tk.BooleanVar(value=saved.get("show_trails", False))
    trail_length_var = tk.IntVar(value=max(1, min(500, int(saved.get("trail_length", 50)))))
    grouping_var = tk.DoubleVar(value=saved.get("group_distance_multiplier", 2.5))
    positions = PositionHistory(root, _TRAIT_LABELS,
                                colour_mode=saved.get("position_colour", "Traits / types"),
                                order_by=saved.get("position_order_by", "Agent ID"),
                                view=saved.get("position_view", "Spatial leadership"),
                                on_change=lambda: _save_settings(_collect_settings()))
    layout.setdefault("positions", dict(visible=False))
    _place(positions.window, "positions", f"800x{min(650, screen_h - 120)}+{sim_x}+80")
    positions.window.minsize(460, 220)
    show_positions_var = tk.BooleanVar(value=positions.window.state() != "withdrawn")
    position_colours = {}
    trails = {}
    info_history = deque(maxlen=180)
    type_cache = {}
    advanced_speed_open = {}
    trait_controls, trait_baselines, trait_previews = {}, {}, {}
    zone_limit_cache = {}
    traits_dirty = set()
    cluster_state = dict(last_time=-float("inf"), groups=0, isolated=0, link_distance=0., milling={})
    type_vars, type_checks = {}, {}
    steps_per_frame = tk.IntVar(value=1)
    width_var, height_var = tk.StringVar(value=str(width)), tk.StringVar(value=str(height))
    view = [1., 0., 0.]
    camera = dict(zoom=1., x=.5, y=.5, drag=None)
    display_size_var = tk.DoubleVar(value=saved.get("display_size", 1.))
    body_style = tk.StringVar(value=saved.get("body_style", "Triangle"))
    lock_var = tk.BooleanVar(value=saved.get("lock_arena_size", False))
    record_var = tk.BooleanVar(value=False)
    recording_status = tk.StringVar(value="Not recording — 0 samples")
    recording = tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8")
    writer = csv.writer(recording)
    parameter_recording = tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8")
    parameter_writer = csv.writer(parameter_recording)
    previous_parameters = {}
    trait_recording = tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8")
    trait_writer = csv.writer(trait_recording)
    mouse_recording = tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8")
    mouse_writer = csv.writer(mouse_recording)
    mouse_trajectory = MouseTrajectory()
    recorded_traits = {}
    columns = ("sample", "step", "id", "group", "x", "y", "heading_deg", "dx", "dy", "width", "height")
    state.update(canvas_size=None, samples=0, recorded_rows=0)

    def _record_state(force=False):
        if not record_var.get() and not force:
            return
        _log_parameter_changes()
        _record_traits()
        sample = state["samples"]
        count = 0
        for group, agents in model.groups.items():
            for a in agents:
                writer.writerow((sample, state["step"], a.unique_id, group, a.x, a.y,
                                 math.degrees(math.atan2(a.dy, a.dx) if math.hypot(a.dx, a.dy) > 1e-10 else a.heading) % 360,
                                 a.dx, a.dy, model.env.width, model.env.height))
                count += 1
        mouse_row = mouse_trajectory.sample(_mouse_position(), state['step'])
        mouse_writer.writerow((sample, state['step'], 'mouse', 'mouse_predator', *mouse_row,
                               model.env.width, model.env.height))
        if mouse_row[0]:
            count += 1
        state["samples"] += 1
        state["recorded_rows"] += count
        _refresh_recording_controls()

    def _reset_recording():
        recording.seek(0)
        recording.truncate()
        writer.writerow(columns)
        state.update(samples=0, recorded_rows=0)
        mouse_recording.seek(0)
        mouse_recording.truncate()
        mouse_writer.writerow(('sample', 'step', 'id', 'group', 'active', 'x', 'y', 'heading_deg', 'dx', 'dy', 'width', 'height'))
        mouse_trajectory.reset()
        parameter_recording.seek(0)
        parameter_recording.truncate()
        parameter_writer.writerow(("step", "applies_from_step", "sample", "event", "parameter", "old_value", "new_value"))
        previous_parameters.clear()
        trait_recording.seek(0)
        trait_recording.truncate()
        trait_writer.writerow(("step", "applies_from_step", "sample", "event", "id", "group", "parameter", "value"))
        recorded_traits.clear()
        traits_dirty.clear()
        _record_state()
        _refresh_recording_controls()

    selected_model = saved.get("interaction_model", all_types[0].interaction_model)
    if selected_model not in {"zonal", "voronoi"}:
        root.destroy()
        raise ValueError("interaction_model must be 'zonal' or 'voronoi'")
    model_var = tk.StringVar(value=selected_model)
    for t in all_types:
        t.interaction_model = selected_model

    def _scrollable(window):
        outer = tk.Frame(window)
        outer.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(outer)
        area = tk.Canvas(outer, highlightthickness=0, yscrollcommand=scroll.set)
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=area.yview)
        def _scroll_visibility(first, last):
            scroll.set(first, last)
            if float(first) <= 0 and float(last) >= 1:
                scroll.pack_forget()
            elif not scroll.winfo_manager():
                scroll.pack(side=tk.RIGHT, fill=tk.Y, before=area)
        area.config(yscrollcommand=_scroll_visibility)
        frame = tk.Frame(area, padx=8, pady=6)
        item = area.create_window(0, 0, window=frame, anchor="nw")
        frame.bind("<Configure>", lambda _: area.config(scrollregion=area.bbox("all")))
        area.bind("<Configure>", lambda event: area.itemconfig(item, width=event.width))

        bind_wheel(window, lambda event: area.yview_scroll(wheel_units(event), "units"))
        return frame

    top = _scrollable(controls)
    parameter_top = _scrollable(parameters)

    def _redraw_arena():
        view[:] = _arena_transform(model.env.width, model.env.height,
                                   canvas.winfo_width(), canvas.winfo_height())
        view[0] *= camera["zoom"]
        for axis, dimension, screen in (("x", model.env.width, canvas.winfo_width()),
                                         ("y", model.env.height, canvas.winfo_height())):
            extent = dimension * view[0]
            offset = (screen - extent) / 2 if extent <= screen else min(0., max(screen - extent, screen / 2 - camera[axis] * extent))
            view[1 if axis == "x" else 2] = offset
            camera[axis] = (screen / 2 - offset) / extent
            bar = horizontal if axis == "x" else vertical
            bar.set(max(0., -offset / extent), min(1., (screen - offset) / extent))
        scale, ox, oy = view
        canvas.delete("arena")
        canvas.delete("walls")
        canvas.create_rectangle(ox, oy, ox + model.env.width * scale, oy + model.env.height * scale,
                                fill="white", outline="#444444", width=2,
                                dash=() if model.env.bounded else (4, 4), tags="arena")
        draw_walls(canvas, model.env, view)
        canvas.tag_lower("walls")
        canvas.tag_lower("arena")

    zone_settings = copy.deepcopy(saved.get('zone_panel', {}))
    zone_panels = {}
    show_zones_var = tk.BooleanVar(value=zone_settings.get('visible', True))
    # The standalone window's geometry is obsolete; diagrams now belong to types.
    layout.pop('zones', None)

    def _zone_settings():
        preferences = copy.deepcopy(zone_settings.get('types', {}))
        preferences.update({name: panel.settings() for name, panel in zone_panels.items()})
        return dict(visible=show_zones_var.get(), types=preferences)

    def _toggle_zones():
        for panel in zone_panels.values():
            if show_zones_var.get():
                panel.frame.pack(fill=tk.X, pady=(0, 8))
                panel.render(force=True)
            else:
                panel.frame.pack_forget()
        _save_settings(_collect_settings())

    def _draw_edges():
        canvas.delete("edges")
        if not show_edges_var.get():
            return
        pos = np.array([[a.x, a.y] for a in model.all_agents], dtype=float)
        if len(pos) < 2:
            return
        edges = {(0, 1)} if len(pos) == 2 else set()
        if len(pos) >= 3:
            try:
                triangles = Delaunay(pos).simplices
            except QhullError:
                return
            for triangle in triangles:
                for i in range(3):
                    edges.add(tuple(sorted((triangle[i], triangle[(i + 1) % 3]))))
        scale, ox, oy = view
        for i, j in edges:
            size = None if model.env.bounded else (model.env.width, model.env.height)
            for a, b, _ in wrapped_segments(pos[i], pos[j], size):
                canvas.create_line(ox + a[0] * scale, oy + a[1] * scale,
                                   ox + b[0] * scale, oy + b[1] * scale,
                                   fill="#bbbbbb", tags="edges")

    def _update_trails():
        if not show_trails_var.get():
            trails.clear()
            return
        alive = {agent.unique_id for agent in model.all_agents}
        for uid in set(trails) - alive:
            del trails[uid]
        for agent in model.all_agents:
            trails.setdefault(agent.unique_id, deque(maxlen=trail_length_var.get())).append((agent.x, agent.y))

    def _toggle_trails():
        trails.clear()
        _update_trails()
        draw()

    def _set_grouping_multiplier(value):
        positions.clear()
        position_colours.clear()
        cluster_state["last_time"] = -float("inf")
        _refresh_info(force=True)

    def _set_trail_length(value):
        length = int(float(value))
        for uid, points in list(trails.items()):
            trails[uid] = deque(points, maxlen=length)
        draw()

    def _draw_trails():
        canvas.delete("trails")
        if not show_trails_var.get():
            return
        scale, ox, oy = view
        for agent in model.all_agents:
            points = trails.get(agent.unique_id, ())
            rgb = canvas.winfo_rgb(agent.color or agent.type.color)
            colour = "#" + "".join(f"{round((channel / 257 + 255) / 2):02x}" for channel in rgb)
            for (x0, y0), (x1, y1) in zip(points, list(points)[1:]):
                # Split periodic wraps instead of drawing a line across the arena.
                if not model.env.bounded and (abs(x1 - x0) > model.env.width / 2 or
                                              abs(y1 - y0) > model.env.height / 2):
                    continue
                canvas.create_line(ox + x0 * scale, oy + y0 * scale,
                                   ox + x1 * scale, oy + y1 * scale,
                                   fill=colour, tags="trails")

    def _update_positions(force=False):
        if positions.window.state() == "withdrawn" and positions.colour_var.get() == "Traits / types":
            position_colours.clear()
            return
        if positions.update(model.all_agents, model.env, grouping_var.get(), state["step"], force=force):
            position_colours.clear()
            position_colours.update(positions.colours())

    def _show_positions():
        _show(positions.window)
        show_positions_var.set(True)
        _update_positions(force=True)

    def _change_position_colour(_value=None):
        _update_positions(force=True)
        _save_settings(_collect_settings())
        draw()

    def _sync_zone_limits():
        for t in all_types:
            cap = visual_range_var.get() * t.visual_range_multiplier
            if zone_limit_cache.get(t.name) != cap:
                t.constrain_zones(cap, model.groups[t.name])
                zone_limit_cache[t.name] = cap
                traits_dirty.add(t.name)
            controls_for_type = trait_controls.get(t.name, {})
            for key, low, high in (("r_r", 0., t.r_o), ("r_o", t.r_r, t.r_a), ("r_a", t.r_o, cap)):
                item = controls_for_type.get(key)
                if item and "scale" in item:
                    widget = item['scale']
                    if float(widget.cget('from')) != low or float(widget.cget('to')) != high:
                        widget.config(from_=low, to=high)
                    if item['var'].get() != getattr(t, key):
                        item['var'].set(getattr(t, key))
                    trait_baselines[t.name][key] = getattr(t, key)

    def _mouse_position():
        if not mouse_predator_var.get() or state['mouse_pointer'] is None:
            return None
        if wall_editor.mode.get() != 'Off' or camera['drag'] is not None:
            return None
        return arena_pointer(state['mouse_pointer'], view, (model.env.width, model.env.height),
                             (canvas.winfo_width(), canvas.winfo_height()))

    def _draw_mouse_predator():
        canvas.delete('mouse-predator')
        point = _mouse_position()
        if point is not None:
            scale, ox, oy = view
            x, y = ox + point[0]*scale, oy + point[1]*scale
            canvas.create_oval(x-7, y-7, x+7, y+7, outline='#c63838', width=2, tags='mouse-predator')
            canvas.create_line(x-10, y, x+10, y, fill='#c63838', tags='mouse-predator')
            canvas.create_line(x, y-10, x, y+10, fill='#c63838', tags='mouse-predator')

    def _mouse_move(event):
        state['mouse_pointer'] = (event.x, event.y)
        _draw_mouse_predator()

    def _mouse_leave(_event):
        state['mouse_pointer'] = None
        canvas.delete('mouse-predator')

    def _toggle_mouse_predator():
        if mouse_predator_var.get():
            root.lift()
            root.focus_force()
            canvas.focus_set()
        _draw_mouse_predator()
        _log_parameter_changes()

    def draw():
        if not popup_active(root):
            _sync_zone_limits()
            for panel in zone_panels.values():
                panel.render()
            _update_positions()
        canvas.delete("agents")
        _draw_edges()
        _draw_trails()
        visual_network.draw(model.all_agents, model.env, view, show_visual_var.get(),
                            body_style.get(), fov_var.get(),
                            tuple(visual_range_var.get() * a.type.visual_range_multiplier for a in model.all_agents),
                            visual_fraction_var.get(), state["step"])
        scale, ox, oy = view
        for agent in model.all_agents:
            if not math.isfinite(agent.x) or not math.isfinite(agent.y):
                continue
            angle = math.atan2(agent.dy, agent.dx) if math.hypot(agent.dx, agent.dy) > 1e-10 else agent.heading
            x, y = ox + agent.x * scale, oy + agent.y * scale
            length = body_size(agent.type) * scale * max(1., display_size_var.get())
            if x + length < 0 or y + length < 0 or x - length > canvas.winfo_width() or y - length > canvas.winfo_height():
                continue
            geometry = body_geometry(agent.x, agent.y, angle, body_size(agent.type), body_style.get(),
                                     display_scale=display_size_var.get())
            colour = position_colours.get(agent.unique_id, agent.color or agent.type.color)
            for kind, values in geometry.items():
                coords = [value * scale + (ox if i % 2 == 0 else oy) for i, value in enumerate(values)]
                if kind == "polygon":
                    canvas.create_polygon(*coords, fill=colour, outline="", tags="agents")
                elif kind == "circle":
                    canvas.create_oval(*coords, fill=colour, outline="", tags="agents")
                else:
                    canvas.create_line(*coords, fill=colour, width=max(1., scale) + 1., tags="agents")

        canvas.tag_raise("visual-network")
        canvas.tag_raise("agents")
        _draw_distance_scale()
        _draw_mouse_predator()

    def _draw_distance_scale():
        canvas.delete("distance-scale")
        if not show_scale_var.get():
            return
        x, y = 18, canvas.winfo_height() - 22
        width = 100 * view[0]
        canvas.create_rectangle(x - 7, y - 12, x + width + 7, y + 12,
                                fill="white", outline="", tags="distance-scale")
        canvas.create_line(x, y, x + width, y, fill="#333333", width=2, tags="distance-scale")
        for fraction in (0, .25, .5, .75, 1):
            tick = x + width * fraction
            canvas.create_line(tick, y - 5, tick, y + 5, fill="#333333", tags="distance-scale")
        fps = canvas.create_text(18, 18, text=f"FPS: {state['fps']:.0f}", anchor="nw",
                                 fill="#333333", font=("TkDefaultFont", 10), tags="distance-scale")
        left, top, right, bottom = canvas.bbox(fps)
        background = canvas.create_rectangle(left - 6, top - 4, right + 6, bottom + 4,
                                             fill="white", outline="", tags="distance-scale")
        canvas.tag_lower(background, fps)
        canvas.tag_raise("distance-scale")

    visual_network = VisualNetwork(canvas, draw)

    def _refresh_info(force=False):
        if popup_active(root):
            return
        now = time.perf_counter()
        if not force and now - state["last_info"] < .15:
            return
        state["last_info"] = now
        _update_positions(force=force)
        result = metrics.compute(model.groups, model.env)
        info_canvas.delete("all")
        if now-cluster_state['last_time']>=.5 or force:
            all_agents = model.all_agents
            clustering = metrics.spatial_groups(all_agents, model.env, multiplier=grouping_var.get(), return_labels=True)
            cluster_state.update({key:clustering[key] for key in ('groups','isolated','link_distance')}, last_time=now,
                                 milling=metrics.milling_by_type(all_agents,model.env,clustering['labels']))
        values = {name:(result[f'{name}_polarisation'],result[f'{name}_nnd'],result[f'{name}_speed'],
                        cluster_state['milling'].get(name,float('nan'))) for name in model.groups}
        values['spatial']=(cluster_state['groups'],)
        sample=(state['step'],values)
        if info_history and info_history[-1][0]==state['step']:
            info_history[-1]=sample
        else:
            info_history.append(sample)
        info_canvas.create_text(12,12,anchor='nw',text=f"Step {state['step']:,}    {state['fps']:.0f} fps",
                                font=('TkDefaultFont',16,'bold'))
        y=48
        for name, atype in model.agent_types.items():
            pol, nnd, speed, milling = values[name]
            milling_text = f"{milling:.2f}" if math.isfinite(milling) else "—"
            info_canvas.create_text(12, y, anchor="nw", fill=atype.color,
                                    text=f"{name}:  P {pol:.2f}   NND {nnd:.1f}   Speed {speed:.2f}   M {milling_text}",
                                    font=("TkDefaultFont", 11))
            y += 25
        plot_width = max(220, info_canvas.winfo_width() - 65)
        first_step = info_history[0][0]
        last_step = max(first_step + 1, info_history[-1][0])
        for metric, title in enumerate(("Polarisation", "Cohesion (nearest-neighbour distance)", "Mean speed", "Milling (within spatial groups)")):
            info_canvas.create_text(12, y, anchor="nw", text=title, font=("TkDefaultFont", 12, "bold"), tags="milling-title" if metric==3 else "metric-title")
            y += 28
            left, bottom, height = 45, y + 85, 85
            maximum = 1. if metric in (0,3) else max(
                (v[name][metric] for _, v in info_history for name in model.groups if name in v), default=1.)
            maximum = max(maximum, 1e-6)
            info_canvas.create_line(left, y, left, bottom, left + plot_width, bottom, fill="#bbbbbb")
            info_canvas.create_text(left - 5, y, anchor="e", text=f"{maximum:.2g}", fill="#777777")
            info_canvas.create_text(left - 5, bottom, anchor="e", text="0", fill="#777777")
            for name, atype in model.agent_types.items():
                segment = []
                for step, observed in info_history:
                    if name not in observed or not math.isfinite(observed[name][metric]):
                        if len(segment) >= 4:
                            info_canvas.create_line(*segment, fill=atype.color, width=2, tags="milling" if metric==3 else "history")
                        segment = []
                        continue
                    segment.extend((left + plot_width * (step - first_step) / (last_step - first_step),
                                    bottom - height * observed[name][metric] / maximum))
                if len(segment) >= 4:
                    info_canvas.create_line(*segment, fill=atype.color, width=2, tags="milling" if metric==3 else "history")
            info_canvas.create_text(left, bottom + 5, anchor="nw", text=str(first_step), fill="#777777")
            info_canvas.create_text(left + plot_width, bottom + 5, anchor="ne",
                                    text=str(info_history[-1][0]), fill="#777777")
            y = bottom + 32
        info_canvas.create_text(12, y, anchor="nw", text="Spatial groups (all types, ≥2 agents)",
                                font=("TkDefaultFont", 12, "bold"))
        y += 28
        group_max = max(1, max(v.get("spatial", (0,))[0] for _, v in info_history))
        left, bottom = 45, y + 85
        info_canvas.create_line(left, y, left, bottom, left + plot_width, bottom, fill="#bbbbbb")
        info_canvas.create_text(left - 5, y, anchor="e", text=str(group_max), fill="#777777")
        info_canvas.create_text(left - 5, bottom, anchor="e", text="0", fill="#777777")
        points = []
        for step, observed in info_history:
            points.extend((left + plot_width * (step - first_step) / (last_step - first_step),
                           bottom - 85 * observed.get("spatial", (0,))[0] / group_max))
        if len(points) >= 4:
            info_canvas.create_line(*points, fill="#444444", width=2, tags="clusters")
        info_canvas.create_text(left, bottom + 5, anchor="nw", text=str(first_step), fill="#777777")
        info_canvas.create_text(left + plot_width, bottom + 5, anchor="ne", text=str(info_history[-1][0]), fill="#777777")
        y = bottom + 28
        info_canvas.create_text(12, y, anchor="nw",
                                text=f"Groups: {cluster_state['groups']}   Isolated: {cluster_state['isolated']}   Link: {cluster_state['link_distance']:.1f}",
                                fill="#777777")
        y += 30
        # Keep colour legends for heterogeneous types without repeating population counts.
        for name, atype in model.agent_types.items():
            if not atype.has_colorbar:
                continue
            label = getattr(atype, "_color_label", None) or atype.color_by
            info_canvas.create_text(12, y, anchor="nw", text=f"{name}: {label.replace('_', ' ')}")
            y += 20
            for i, colour in enumerate(atype._colorbar):
                info_canvas.create_rectangle(12 + i * 5, y, 17 + i * 5, y + 10, fill=colour, outline="")
            y += 16
            factor = 180 / math.pi if atype.color_by in {"max_turn", "noise"} else 1.
            info_canvas.create_text(12, y, anchor="nw", text=f"{atype._dist_min * factor:.3g}")
            info_canvas.create_text(252, y, anchor="ne", text=f"{atype._dist_max * factor:.3g}")
            y += 25
        info_canvas.config(scrollregion=(0, 0, max(300, info_canvas.winfo_width()), y + 12))

    def _cancel_frame():
        if state["timer"] is not None:
            root.after_cancel(state["timer"])
            state["timer"] = None

    def _schedule(delay=1):
        if state["active"] and not state["moving"] and state["timer"] is None:
            state["timer"] = root.after(delay, loop)

    def loop():
        state["timer"] = None
        if not state["active"] or state["moving"] or state["closing"]:
            return
        if state["remaining"] == 0:
            state["remaining"] = steps_per_frame.get()
        deadline = time.perf_counter() + .012
        kw = dict(capture_distance=capture_distance_var.get(), desired_speed=speed_var.get(), fov_deg=fov_var.get(), visual_range=visual_range_var.get(), repulsion_fov_deg=repulsion_fov_var.get())
        try:
            while state["remaining"] > 0:
                _log_parameter_changes()
                point = _mouse_position()
                model.step(**kw, external_threats=[point] if point is not None else None)
                _update_trails()
                state["step"] += 1
                state["remaining"] -= 1
                _record_state()
                if rows is not None:
                    rows.append({"step": state["step"], "width": model.env.width, "height": model.env.height,
                                 **metrics.compute(model.groups, model.env)})
                if time.perf_counter() >= deadline:
                    break
        except (ValueError, RuntimeError) as exc:
            stop_sim()
            messagebox.showerror("Simulation stopped", str(exc), parent=controls)
            return
        if state["remaining"]:
            _schedule()
            return
        now = time.perf_counter()
        dt = now - state["last_frame"]
        state["last_frame"] = now
        if dt > 0:
            state["fps"] = .8 * state["fps"] + .2 / dt
        draw()
        _refresh_info()
        _schedule(max(1, int(16 - (time.perf_counter() - now) * 1000)))

    def start_sim():
        state["active"] = True
        state["last_frame"] = time.perf_counter()
        status.set("Resizing / moving" if state["moving"] else "Running")
        _schedule()

    def stop_sim():
        state["active"] = False
        state["remaining"] = 0
        state["fps"] = 0.
        _cancel_frame()
        status.set("Paused")
        draw()
        _refresh_info(force=True)

    def _finish_move():
        if state["settle"] is not None:
            root.after_cancel(state["settle"])
            state["settle"] = None
        if state["closing"]:
            return
        if popup_active(root):
            state['settle']=root.after(100,_finish_move)
            return
        size = (canvas.winfo_width(), canvas.winfo_height())
        if min(size) < 2:
            state["moving"] = False
            return
        previous = state["canvas_size"]
        if previous is not None and size != previous and not lock_var.get():
            previous_scale = min(previous[0] / model.env.width, previous[1] / model.env.height)
            w = max(2, round(size[0] / previous_scale))
            h = max(2, round(size[1] / previous_scale))
            camera.update(zoom=1., x=.5, y=.5)
            try:
                environment = Environment(w, h, bounded=model.env.bounded, mask=mask,
                                          avoid_radius=wall_ar_var.get(), avoid_strength=wall_as_var.get(),
                                          thigmotaxis=wall_thig_var.get(), wall_shapes=model.env.wall_shapes, spawn_radius=model.env.spawn_radius)
                agents = model.all_agents
                positions = np.array([[a.x, a.y] for a in agents], dtype=float).reshape(-1, 2)
                positions *= [w / model.env.width, h / model.env.height]
                environment.project_free(positions)
                for a, (x, y) in zip(agents, positions):
                    a.x, a.y = float(x), float(y)
                model.env = environment
                trails.clear()
                width_var.set(str(w))
                height_var.set(str(h))
                _record_state()
            except (ValueError, OSError, RuntimeError) as exc:
                state["active"] = False
                messagebox.showerror("Cannot resize arena", str(exc), parent=controls)
        state["canvas_size"] = size
        state["moving"] = False
        state["last_frame"] = time.perf_counter()
        status.set("Running" if state["active"] else "Paused")
        _redraw_arena()
        draw()
        _refresh_info(force=True)
        _save_settings(_collect_settings())
        _schedule()

    def _on_window_change(event):
        if state["closing"] or event.widget not in (root, canvas, controls, parameters, info, positions.window, *panels.values()):
            return
        if event.widget is canvas and state["canvas_size"] is None and min(event.width, event.height) > 1:
            state["canvas_size"] = (event.width, event.height)
        state["moving"] = True
        _cancel_frame()
        status.set("Resizing / moving" if state["active"] else "Paused")
        if state["settle"] is not None:
            root.after_cancel(state["settle"])
        state["settle"] = root.after(180, _finish_move)

    def _on_release(event):
        if event.widget not in (root,canvas,controls,parameters,info,positions.window,*panels.values()) or popup_active(root):
            return
        if not state["closing"] and state["moving"]:
            _finish_move()

    def _set_group_count(name, count):
        agents = model.groups[name]
        if count > len(agents):
            model.init_agents(name, count - len(agents))
        elif count < len(agents):
            keep = set(model.rng.choice(len(agents), size=count, replace=False).tolist())
            model.groups[name] = [agent for i, agent in enumerate(agents) if i in keep]
        model.agent_types[name].update_colors(model.groups[name])
        if name in trait_previews: trait_previews[name]["preview"]()
        draw()
        _refresh_info(force=True)

    def _nudge(var, delta, lo, hi, cb=None):
        v = max(lo, min(hi, var.get() + delta))
        var.set(v)
        if cb:
            cb(v)
        _log_parameter_changes()

    def _slider_row(parent, label, var, lo, hi, res, cb=None):
        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        label_options = {"textvariable": label} if isinstance(label, tk.StringVar) else {"text": label}
        tk.Label(row, **label_options, anchor=tk.W, justify=tk.LEFT,
                 wraplength=110, width=15).pack(side=tk.LEFT)
        slider_area = tk.Frame(row)
        slider_area.pack(side=tk.LEFT, fill=tk.X, expand=True)
        slider_area.columnconfigure(1, weight=1)
        tk.Label(slider_area, textvariable=var).grid(row=0, column=1)
        tk.Button(slider_area, text="◀", width=2,
                  command=lambda: _nudge(var, -res, float(scale_widget.cget('from')), float(scale_widget.cget('to')), cb)).grid(row=1, column=0)
        def changed(value):
            if cb:
                cb(value)
            _log_parameter_changes()
        scale_widget = tk.Scale(slider_area, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                                variable=var, length=80, showvalue=False, command=changed)
        scale_widget.grid(row=1, column=1, sticky="ew")
        tk.Button(slider_area, text="▶", width=2,
                  command=lambda: _nudge(var, +res, float(scale_widget.cget('from')), float(scale_widget.cget('to')), cb)).grid(row=1, column=2)
        tk.Frame(parent, height=1, bg="#cccccc").pack(fill=tk.X, pady=2)
        return scale_widget

    def _trait_changed(atype, field, value, callback=None):
        controls_for_type = trait_controls[atype.name]
        old = trait_baselines[atype.name].copy()
        old_specs = copy.deepcopy(atype._distributions)
        if callback:
            callback(value)
        else:
            setattr(atype, field, float(value))
        changed = {key for key in old if getattr(atype, key) != old[key]}
        if not changed:
            return
        try:
            for key in changed:
                spec = atype._distributions.get(key)
                if not spec:
                    continue
                centre = getattr(atype, key)
                kind = spec["_dist"]
                if kind == "uniform":
                    half = (spec.get("high", old[key] * 1.5) - spec.get("low", old[key] * .5)) / 2
                    lower = {"preferred_distance": 1e-6, "max_speed_scale": 1.}.get(key, 0.)
                    upper = {"social_speed": 1., "max_turn": math.pi}.get(key, float("inf"))
                    half = max(0., min(half, centre - lower, upper - centre))
                    spec.update(low=centre - half, high=centre + half)
                elif kind == "normal":
                    bounds = spec.get("_clip")
                    if key in {"r_r", "r_o", "r_a"} and bounds:
                        bounds = [min(centre, bounds[0]) if bounds[0] is not None else None,
                                  max(centre, bounds[1]) if bounds[1] is not None else None]
                        spec['_clip'] = bounds
                    if bounds and not (bounds[0] if bounds[0] is not None else -np.inf) <= centre <= (bounds[1] if bounds[1] is not None else np.inf):
                        raise ValueError("Keep the mean inside its distribution bounds, or edit those bounds first.")
                    spec["mean"] = centre
                elif kind == "lognormal":
                    spec["mean"] = math.log(max(centre, 1e-9))
                elif kind == "bimodal":
                    spec["means"] = [v + centre - old[key] for v in spec.get("means", [old[key] * .6, old[key] * 1.4])]
            if changed <= {"r_r", "r_o", "r_a"}:
                agents_now = model.groups[atype.name]
                samples = atype._sample(len(agents_now), rng=model.rng, parameters=changed)
                for i, agent in enumerate(agents_now):
                    for key in changed:
                        if key in samples:
                            agent.individual[key] = float(samples[key][i])
                        else:
                            agent.individual.pop(key, None)
                atype.constrain_zones(visual_range_var.get() * atype.visual_range_multiplier, agents_now)
            else:
                atype.apply_variation(model.groups[atype.name], changed, rng=model.rng)
        except (ValueError, TypeError) as exc:
            atype._distributions = old_specs
            for key, previous in old.items():
                setattr(atype, key, previous)
                controls_for_type[key]["var"].set(previous / controls_for_type[key]["factor"])
            messagebox.showerror("Cannot change individual values", str(exc), parent=panels.get(atype.name, controls))
            return
        trait_baselines[atype.name] = {key: getattr(atype, key) for key in old}
        _refresh_trait_titles(atype)
        traits_dirty.add(atype.name)
        _record_traits()
        if atype.name in trait_previews:
            trait_previews[atype.name]["reload"]()
        draw()
        _refresh_info(force=True)

    def _refresh_trait_titles(atype):
        for key, item in trait_controls[atype.name].items():
            item["title"].set(item["label"].get() + (" · mean" if key in atype._distributions else ""))

    def _variation_editor(parent, colour_row, atype):
        fields = trait_controls[atype.name]
        labels = {_TRAIT_LABELS[key]: key for key in fields}
        colour_var = tk.StringVar(value=_TRAIT_LABELS.get(atype.color_by, "Type colour"))
        if atype.color_by:
            atype._color_label = _TRAIT_LABELS.get(atype.color_by, atype.color_by)
        tk.Label(colour_row, text="Colour by:").pack(side=tk.LEFT)
        def colour_changed(label):
            atype.color_by = labels.get(label)
            atype._color_label = label if atype.color_by else None
            atype.update_colors(model.groups[atype.name])
            _log_parameter_changes()
            draw()
            _refresh_info(force=True)
        choice(colour_row, colour_var, "Type colour", *labels, command=colour_changed).pack(side=tk.LEFT)
        tk.Frame(parent, height=1, bg="#aaaaaa").pack(fill=tk.X, pady=(8, 4))
        tk.Label(parent, text="Individual variation", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill=tk.X)
        parameter_var = tk.StringVar(value="Speed multiplier")
        choice(parent, parameter_var, *labels, command=lambda _: reload_editor()).pack(fill=tk.X)
        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        tk.Label(row, text="Distribution:").pack(side=tk.LEFT)
        mode_var = tk.StringVar(value="Fixed")
        choice(row, mode_var, "Fixed", "Normal", "Uniform", command=lambda _: show_fields()).pack(side=tk.LEFT)
        detail = tk.Frame(parent)
        std_var, low_var, high_var = (tk.StringVar() for _ in range(3))
        std_row = tk.Frame(detail)
        tk.Label(std_row, text="Spread (SD)", width=13, anchor="w").pack(side=tk.LEFT)
        tk.Entry(std_row, textvariable=std_var, width=10).pack(side=tk.LEFT)
        for title, var in (("Minimum", low_var), ("Maximum", high_var)):
            line = tk.Frame(detail)
            line.pack(fill=tk.X)
            tk.Label(line, text=title, width=13, anchor="w").pack(side=tk.LEFT)
            tk.Entry(line, textvariable=var, width=10).pack(side=tk.LEFT)
        note = tk.StringVar()
        tk.Label(parent, textvariable=note, fg="#777777", wraplength=290, justify=tk.LEFT).pack(anchor="w")
        preview = tk.Canvas(parent, height=85, highlightthickness=0, bg="white")
        preview.pack(fill=tk.X)
        message = tk.StringVar()
        tk.Label(parent, textvariable=message, fg="#777777", wraplength=290, justify=tk.LEFT).pack(anchor="w")

        def physical_bounds(key):
            lower, upper = 0., float("inf")
            if key in {"r_r", "r_o", "r_a"}: upper = visual_range_var.get() * atype.visual_range_multiplier
            if key == "social_speed": upper = 1.
            if key == "max_turn": upper = math.pi
            if key == "preferred_distance": lower = 1e-6
            if key == "max_speed_scale": lower = 1.
            agents = model.groups[atype.name]
            if key in {"r_o", "r_a"}:
                other = "r_r" if key == "r_o" else "r_o"
                lower = max([lower] + [a.individual.get(other, getattr(atype, other)) for a in agents])
            if key in {"r_r", "r_o"}:
                other = "r_o" if key == "r_r" else "r_a"
                upper = min([upper] + [a.individual.get(other, getattr(atype, other)) for a in agents])
            return lower, upper

        def proposed():
            key = labels[parameter_var.get()]
            factor = fields[key]["factor"]
            centre = getattr(atype, key)
            low, high = physical_bounds(key)
            if mode_var.get() != "Fixed":
                if low_var.get().strip():
                    entered = float(low_var.get()) * factor
                    if not math.isfinite(entered): raise ValueError("Minimum must be finite or blank")
                    low = max(low, entered)
                if high_var.get().strip():
                    entered = float(high_var.get()) * factor
                    if not math.isfinite(entered): raise ValueError("Maximum must be finite or blank")
                    high = min(high, entered)
            if not low <= high or not math.isfinite(low):
                raise ValueError("Minimum must not exceed maximum or the valid zone limits")
            if mode_var.get() == "Normal":
                sd = float(std_var.get()) * factor
                if not math.isfinite(sd) or sd < 0 or not low <= centre <= high:
                    raise ValueError("Use a nonnegative SD and keep the mean inside the bounds")
                return key, centre, dict(dist="normal", mean=centre, std=sd, clip=(low, high if math.isfinite(high) else None), truncate=True)
            if mode_var.get() == "Uniform":
                if not math.isfinite(high):
                    raise ValueError("Uniform requires finite minimum and maximum")
                return key, (low + high) / 2, dict(dist="uniform", low=low, high=high)
            if mode_var.get() != "Fixed":
                raise ValueError("Select Fixed, Normal or Uniform to replace this imported distribution")
            return key, centre, None

        def draw_preview(*_):
            preview.delete("all")
            key = labels[parameter_var.get()]
            factor = fields[key]["factor"]
            values = np.array([a.individual.get(key, getattr(atype, key)) / factor for a in model.groups[atype.name]])
            try:
                _, centre, spec = proposed()
                if spec is None:
                    x = np.array([centre / factor])
                    density = np.ones(1)
                elif spec["dist"] == "uniform":
                    x = np.linspace(spec["low"] / factor, spec["high"] / factor, 80)
                    density = np.ones(len(x))
                else:
                    from scipy.stats import truncnorm
                    sd, low, high = spec["std"], spec["clip"][0], spec["clip"][1]
                    high = np.inf if high is None else high
                    if sd == 0 or low == high:
                        x, density = np.array([centre / factor]), np.ones(1)
                    else:
                        start, end = max(low, centre - 3.5 * sd), min(high, centre + 3.5 * sd)
                        x = np.linspace(start / factor, end / factor, 80)
                        density = truncnorm.pdf(x * factor, (low - centre) / sd, (high - centre) / sd, loc=centre, scale=sd)
                minimum = min(x.min(), values.min() if len(values) else x.min())
                maximum = max(x.max(), values.max() if len(values) else x.max())
                if maximum <= minimum:
                    padding = max(abs(minimum) * .1, .1)
                    minimum, maximum = minimum - padding, maximum + padding
                span = maximum - minimum
                width = max(200, preview.winfo_width() - 28)
                if len(values):
                    counts, edges = np.histogram(values, bins=12, range=(minimum, maximum if maximum > minimum else minimum + 1e-6))
                    for count, left, right in zip(counts, edges[:-1], edges[1:]):
                        preview.create_rectangle(12 + (left - minimum) / span * width, 62 - 42 * count / max(counts.max(), 1),
                                                 12 + (right - minimum) / span * width, 62, fill="#dddddd", outline="")
                coords = [v for xx, yy in zip(x, density) for v in
                          (12 + (xx - minimum) / span * width, 62 - 45 * yy / max(density.max(), 1e-12))]
                if len(coords) >= 4:
                    preview.create_line(*coords, fill=atype.color, width=2)
                else:
                    preview.create_line(coords[0], 15, coords[0], 62, fill=atype.color, width=2)
                preview.create_text(12, 68, anchor="nw", text=f"{minimum:.3g}", fill="#777777")
                preview.create_text(12 + width, 68, anchor="ne", text=f"{maximum:.3g}", fill="#777777")
            except (ValueError, TypeError):
                preview.create_text(12, 25, anchor="nw", text="Enter valid bounds and spread", fill="#777777")

        def show_fields():
            if mode_var.get() == "Fixed":
                detail.pack_forget()
                note.set("One value for every agent.")
            else:
                detail.pack(fill=tk.X, before=preview)
                if mode_var.get() == "Normal":
                    std_row.pack(fill=tk.X, before=detail.winfo_children()[1])
                    note.set("Slider = mean; SD sets spread. Bounds are optional.")
                else:
                    std_row.pack_forget()
                    note.set("Uniform uses min/max; applying sets the slider to their midpoint.")
            draw_preview()

        def reload_editor():
            key = labels[parameter_var.get()]
            factor = fields[key]["factor"]
            spec = atype._distributions.get(key)
            centre = getattr(atype, key)
            low, high = physical_bounds(key)
            if spec:
                mode_var.set(spec["_dist"].title())
                bounds = spec.get("_clip")
                if spec["_dist"] == "uniform":
                    low, high = spec.get("low", centre * .5), spec.get("high", centre * 1.5)
                elif bounds:
                    low, high = bounds
                    low, high = (-np.inf if low is None else low, np.inf if high is None else high)
                std_var.set(f"{spec.get('std', max(abs(centre) * .2, .001)) / factor:.5g}")
            else:
                mode_var.set("Fixed")
                std_var.set(f"{max(abs(centre) * .2, .001) / factor:.5g}")
            low_var.set(f"{low / factor:.5g}" if math.isfinite(low) else "")
            high_var.set(f"{high / factor:.5g}" if math.isfinite(high) else "")
            if not spec and not math.isfinite(high):
                high_var.set(f"{(centre + max(abs(centre) * .5, .1)) / factor:.5g}")
            message.set("Grey: current agents. Line: proposed distribution.")
            show_fields()

        def apply():
            old_specs, old_baseline = copy.deepcopy(atype._distributions), trait_baselines[atype.name].copy()
            try:
                key, centre, spec = proposed()
                setattr(atype, key, centre)
                if spec is None:
                    atype._distributions.pop(key, None)
                else:
                    atype.set_distribution(key, color_by=False, **spec)
                atype.apply_variation(model.groups[atype.name], key, rng=model.rng)
            except (ValueError, TypeError) as exc:
                atype._distributions = old_specs
                for field, previous in old_baseline.items(): setattr(atype, field, previous)
                message.set(str(exc))
                return
            fields[key]["var"].set(centre / fields[key]["factor"])
            trait_baselines[atype.name] = {key: getattr(atype, key) for key in fields}
            _refresh_trait_titles(atype)
            traits_dirty.add(atype.name)
            _log_parameter_changes()
            _record_traits()
            _save_settings(_collect_settings())
            draw_preview()
            message.set("Applied to current agents; other traits and positions kept.")
            draw()
            _refresh_info(force=True)

        tk.Button(parent, text="Apply variation", command=apply).pack(anchor="w", pady=3)
        for var in (std_var, low_var, high_var): var.trace_add("write", draw_preview)
        preview.bind("<Configure>", draw_preview)
        trait_previews[atype.name] = dict(reload=reload_editor, preview=draw_preview)
        reload_editor()

    def _build_panel(atype):
        """Create an independently movable, scrollable agent-type window."""
        window = tk.Toplevel(root)
        window.title(f"Agent type {atype.name[1:]}")
        index = all_types.index(atype)
        offset = cascade * (index + 1)
        _place(window, f"agent:{atype.name}",
               f"{right_w}x{max(400, screen_h - 120 - offset)}+{right_x + offset}+{40 + offset}")
        window.protocol("WM_DELETE_WINDOW", lambda: _remove_type(atype.name))
        col = _scrollable(window)
        tk.Label(col, text=f"Agent type {atype.name[1:]}", fg=atype.color,
                 font=("TkDefaultFont", 16, "bold"), anchor="w").pack(fill=tk.X, pady=(0, 8))
        zone_host = tk.Frame(col)
        zone_host.pack(fill=tk.X)
        zone_panels[atype.name] = InteractionZones(zone_host,
            lambda t=atype: ([t], view[0], fov_var.get(), visual_range_var.get(), body_style.get(),
                            display_size_var.get(), repulsion_fov_var.get()),
            settings=zone_settings.get('types', {}).get(atype.name, zone_settings))
        if not show_zones_var.get():
            zone_panels[atype.name].frame.pack_forget()
        colour_row = tk.Frame(col)
        colour_row.pack(fill=tk.X)
        trait_controls[atype.name] = {}
        trait_baselines[atype.name] = {key: getattr(atype, key) for key in _TRAIT_LABELS}
        def trait_slider(parent, label, field, var, low, high, resolution, callback=None, factor=1.):
            base_label = label if isinstance(label, tk.StringVar) else tk.StringVar(value=label)
            title = tk.StringVar()
            trait_controls[atype.name][field] = dict(var=var, label=base_label, title=title, factor=factor)
            base_label.trace_add("write", lambda *_: _refresh_trait_titles(atype))
            _refresh_trait_titles(atype)
            trait_controls[atype.name][field]["scale"] = _slider_row(parent, title, var, low, high, resolution,
                        lambda value: _trait_changed(atype, field, float(value) * factor, callback))
        n_vars[atype.name] = tk.IntVar(value=panel_counts.get(atype.name, len(model.groups[atype.name])))
        zor_lbl = tk.StringVar(value=_IMODEL_LABELS[model_var.get()][0])
        zoo_lbl = tk.StringVar(value=_IMODEL_LABELS[model_var.get()][1])
        zoa_lbl = tk.StringVar(value=_IMODEL_LABELS[model_var.get()][2])
        zone_labels[atype.name] = (zor_lbl, zoo_lbl, zoa_lbl)

        # ── Zone / Voronoi sliders (labels swap with model) ─────────────
        zor_v = tk.DoubleVar(value=atype.r_r)
        trait_slider(col, zor_lbl, "r_r", zor_v, 0, atype.r_o, 1)
        zoo_v = tk.DoubleVar(value=atype.r_o)
        trait_slider(col, zoo_lbl, "r_o", zoo_v, atype.r_r, atype.r_a, 1)
        zoa_v = tk.DoubleVar(value=atype.r_a)
        trait_slider(col, zoa_lbl, "r_a", zoa_v, atype.r_o,
                     visual_range_var.get() * atype.visual_range_multiplier, 1)

        # ── Fixed sliders ────────────────────────────────────────────────
        size_multiplier = tk.DoubleVar(value=atype.size_multiplier)
        def _on_size(value, agent_type=atype):
            agent_type.size_multiplier = float(value)
            _log_parameter_changes()
            draw()
        _slider_row(col, "Body size ×", size_multiplier, .1, max(5., atype.size_multiplier), .1, _on_size)
        visual_multiplier = tk.DoubleVar(value=atype.visual_range_multiplier)
        _slider_row(col, "Detection range ×", visual_multiplier, 0, max(5., atype.visual_range_multiplier), .1,
                    lambda value, t=atype: (setattr(t, "visual_range_multiplier", float(value)), draw()))
        det_v = tk.DoubleVar(value=atype.flee_dist)
        def _on_det(val, t=atype): t.flee_dist = float(val)
        trait_slider(col, "Threat detect range", "flee_dist", det_v, 0, max(400, atype.flee_dist), 10, _on_det)

        ss_v = tk.DoubleVar(value=atype.speed_scale)
        def _on_ss(val, t=atype): t.speed_scale = float(val)
        trait_slider(col, "Speed multiplier", "speed_scale", ss_v, 0, max(5.0, atype.speed_scale), 0.1, _on_ss)

        sv_v = tk.DoubleVar(value=atype.social_speed)
        def _on_sv(val, t=atype): t.social_speed = float(val)
        trait_slider(col, "Social speed", "social_speed", sv_v, 0.0, 1.0, 0.05, _on_sv)
        advanced = tk.Frame(col)
        advanced.pack(fill=tk.X)
        advanced_var = tk.BooleanVar(value=advanced_speed_open.get(atype.name, False))
        advanced_body = tk.Frame(advanced)
        def _toggle_advanced_speed():
            advanced_speed_open[atype.name] = advanced_var.get()
            if advanced_var.get():
                advanced_body.pack(fill=tk.X)
            else:
                advanced_body.pack_forget()
        tk.Checkbutton(advanced, text="Advanced speed settings", variable=advanced_var,
                       command=_toggle_advanced_speed).pack(anchor=tk.W)
        if advanced_var.get():
            advanced_body.pack(fill=tk.X)
        for label, field, low, high, resolution in (
                ("Preferred spacing", "preferred_distance", 1., 150., 1.),
                ("Speed recovery", "speed_relaxation", 0., 1., .05),
                ("Speed noise", "speed_noise", 0., 1., .05),
                ("Max speed ×", "max_speed_scale", 1., 5., .25)):
            var = tk.DoubleVar(value=getattr(atype, field))
            trait_slider(advanced_body, label, field, var, low, max(high, var.get()), resolution,
                        lambda value, field=field, t=atype: setattr(t, field, float(value)))

        mt_v = tk.DoubleVar(value=round(math.degrees(atype.max_turn)))
        trait_slider(col, "Max turn", "max_turn", mt_v, 0, 180, 1, factor=math.pi / 180)

        no_v = tk.DoubleVar(value=round(math.degrees(atype.noise)))
        trait_slider(col, "Heading noise (SD, °)", "noise", no_v, 0, max(90, no_v.get()), 1, factor=math.pi / 180)
        memory_var = tk.DoubleVar(value=atype.heading_noise_memory)
        _slider_row(col, "Noise memory (steps)", memory_var, 0, max(50, atype.heading_noise_memory), 1,
                    lambda value, t=atype: setattr(t, "heading_noise_memory", float(value)))

        _variation_editor(col, colour_row, atype)

        # Directed relationships: steering and removal are separate choices.
        others = [t for t in all_types if t.name != atype.name]
        if others:
            tk.Frame(col, height=1, bg="#999999").pack(fill=tk.X, pady=4)
            tk.Label(col, text="Relationships", font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W)
            for other in others:
                variables = {key: tk.BooleanVar(value=other.name in getattr(atype, key))
                             for key in ('pursues', 'flees', 'captures')}
                def change(key, target=other.name, agent_type=atype, variables=variables):
                    enabled = variables[key].get()
                    if enabled:
                        getattr(agent_type, key).add(target)
                        opposite = {'pursues':'flees', 'flees':'pursues'}.get(key)
                        if opposite:
                            getattr(agent_type, opposite).discard(target)
                            variables[opposite].set(False)
                    else:
                        getattr(agent_type, key).discard(target)
                    _log_parameter_changes()
                    draw()
                for key, label in (('pursues', 'Pursues'), ('flees', 'Flees'), ('captures', 'Captures')):
                    tk.Checkbutton(col, text=f"{label} {other.name}", variable=variables[key],
                                   command=lambda key=key, change=change: change(key)).pack(anchor=tk.W)

        window.bind("<Configure>", _on_window_change)
        window.bind("<ButtonRelease-1>", _on_release)
        return window

    def _show(window):
        show_window(window)
        if window is info:
            root.after_idle(lambda: (_refresh_info(force=True), info_canvas.yview_moveto(0)))
            show_info_var.set(True)
        if window is positions.window:
            show_positions_var.set(True)
        _save_settings(_collect_settings())

    def _remember_windows():
        for key, window in windows.items():
            if window.winfo_exists():
                visible=window.state()!='withdrawn'
                geometry=layout.get(key,{}).get('geometry',window.geometry())
                if visible and window.winfo_ismapped() and window.winfo_width()>=160 and window.winfo_height()>=120:
                    geometry=window.geometry()
                layout[key]=dict(geometry=geometry,visible=visible)
        return dict(layout)

    def _on_model(value):
        for t in all_types:
            t.interaction_model = value
            for label, text in zip(zone_labels.get(t.name, ()), _IMODEL_LABELS[value]):
                label.set(text)
        for panel in zone_panels.values():
            panel.render(force=True)
        _log_parameter_changes()
        _save_settings(_collect_settings())

    def _rebuild_panels():
        panel_counts.update({name: var.get() for name, var in n_vars.items()})
        _remember_windows()
        zone_settings['types'] = _zone_settings()['types']
        for window in panels.values():
            window.destroy()
        zone_panels.clear()
        panels.clear()
        n_vars.clear()
        zone_labels.clear()
        trait_previews.clear()
        trait_controls.clear()
        trait_baselines.clear()
        for t in all_types:
            panels[t.name] = _build_panel(t)
        for name, var in type_vars.items():
            var.set(name in model.groups)
            type_checks[name].config(state=tk.DISABLED if name in model.groups and len(all_types) == 1 else tk.NORMAL)
        for child in population_frame.winfo_children():
            child.destroy()
        for t in all_types:
            _slider_row(population_frame, f"Nr of agents: [{t.name}]", n_vars[t.name], 0, max(500, n_vars[t.name].get()), 1,
                        lambda value, name=t.name: _set_group_count(name, int(float(value))))
        windows_menu.delete(0, "end")
        windows_menu.add_command(label="Controls", command=lambda: _show(controls))
        windows_menu.add_command(label="Information", command=lambda: _show(info))
        windows_menu.add_command(label="Individual positions", command=_show_positions)
        windows_menu.add_command(label="General movement settings", command=lambda: _show(parameters))
        windows_menu.add_separator()
        for name, window in panels.items():
            windows_menu.add_command(label=f"Agents: {name}", command=lambda w=window: _show(w))
        _refresh_info(force=True)
        _log_parameter_changes()
        root.after_idle(_raise_windows)

    def _toggle_type(name):
        if type_vars[name].get():
            _add_type(name)
        else:
            _remove_type(name)

    def _add_type(name):
        if name in model.groups or len(all_types) >= _MAX_TYPES:
            return
        t = type_cache.pop(name, all_types[-1].clone(name))
        if name not in panel_counts:
            t.n_default = n_vars[all_types[-1].name].get()
        else:
            t.n_default = panel_counts[name]
        t.interaction_model = model_var.get()
        t.color = {"A1": "#4c78a8", "A2": "#fc9056", "A3": "#56aa9a"}[name]
        t.pursues.discard(name)
        t.flees.discard(name)
        for key in ('pursues', 'flees', 'captures'):
            getattr(t, key).intersection_update(model.agent_types.keys() - {name})
        all_types.append(t)
        all_types.sort(key=lambda agent_type: agent_type.name)
        model.agent_types[name] = t
        model.groups[name] = []
        model.init_agents(name, t.n_default)
        _rebuild_panels()
        draw()

    def _remove_type(name=None):
        if len(all_types) <= 1:
            if name is not None:
                _hide(panels[name], f"agent:{name}")
            return
        t = next(t for t in all_types if t.name == name) if name is not None else all_types[-1]
        t.n_default = n_vars[t.name].get()
        type_cache[t.name] = t
        all_types.remove(t)
        del model.groups[t.name]
        del model.agent_types[t.name]
        for remaining in all_types:
            remaining.pursues.discard(t.name)
            remaining.flees.discard(t.name)
            remaining.captures.discard(t.name)
        _rebuild_panels()
        windows.pop(f"agent:{t.name}", None)
        layout.pop(f"agent:{t.name}", None)
        _update_trails()
        draw()

    def _raise_windows():
        if state["closing"]:
            return
        # Activate this Tk application once; never leave windows permanently topmost.
        root.lift()
        root.focus_force()
        for window in windows.values():
            if window.winfo_exists() and window.state() != "withdrawn":
                window.lift()
        controls.lift()

    def _collect_settings():
        return dict(mouse_predator=mouse_predator_var.get(), zone_panel=_zone_settings(), show_scale_bar=show_scale_var.get(), show_visual_neighbours=show_visual_var.get(), visual_range=visual_range_var.get(),
                    visual_fraction=visual_fraction_var.get(), display_size=display_size_var.get(), body_style=body_style.get(), seed=seed, desired_speed=speed_var.get(), fov_deg=fov_var.get(), repulsion_fov_deg=repulsion_fov_var.get(), spawn_radius=model.env.spawn_radius, capture_distance=capture_distance_var.get(),
                    bounded=model.env.bounded, width=model.env.width, height=model.env.height,
                    window_width=canvas.winfo_width(), window_height=canvas.winfo_height(),
                    wall_avoid_radius=wall_ar_var.get(), wall_strength=wall_as_var.get(),
                    thigmotaxis=wall_thig_var.get(), show_edges=show_edges_var.get(),
                    show_trails=show_trails_var.get(), trail_length=trail_length_var.get(),
                    group_distance_multiplier=grouping_var.get(), position_colour=positions.colour_var.get(),
                    position_order_by=positions.order_var.get(), position_view=positions.mode_var.get(),
                    mask=str(Path(mask).expanduser().resolve()) if mask is not None else None,
                    wall_shapes=model.env.wall_shapes,
                    window_layout=_remember_windows(), interaction_model=model_var.get(),
                    lock_arena_size=lock_var.get(),
                    agents={t.name: {**t.to_dict(), "n_default": n_vars[t.name].get()} for t in all_types})

    def reset_arena():
        stop_sim()
        model.clear()
        trails.clear()
        record_var.set(False)
        info_history.clear()
        positions.clear()
        position_colours.clear()
        state["step"] = 0
        if rows is not None:
            rows.clear()
        for t in all_types:
            model.init_agents(t.name, n_vars[t.name].get())
        _reset_recording()
        _redraw_arena()
        draw()
        _refresh_info(force=True)

    def _apply_arena_size():
        try:
            w, h = int(width_var.get()), int(height_var.get())
            if w < 2 or h < 2:
                raise ValueError("Width and height must be integers >= 2")
            environment = Environment(w, h, bounded=bounded_var.get(), mask=mask,
                                      avoid_radius=wall_ar_var.get(), avoid_strength=wall_as_var.get(),
                                      thigmotaxis=wall_thig_var.get(), wall_shapes=model.env.wall_shapes, spawn_radius=model.env.spawn_radius)
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Cannot change arena size", str(exc), parent=controls)
            return
        model.env = environment
        reset_arena()

    def _on_bounded():
        model.env.bounded = bounded_var.get()
        _log_parameter_changes()
        _wall_availability()
        _redraw_arena()
        draw()
        _refresh_info(force=True)

    def exit_sim():
        if state["closing"]:
            return
        state["closing"] = True
        _cancel_frame()
        if state["settle"] is not None:
            root.after_cancel(state["settle"])
            state["settle"] = None
        try:
            _save_settings(_collect_settings())
        finally:
            # Let the current Tk button-release binding finish before destroying
            # its registered Python commands and windows.
            root.after_idle(root.quit)

    def _refresh_recording_controls():
        active = record_var.get()
        recording_status.set(f"{'Recording' if active else 'Not recording'} — {state['samples']:,} samples")
        start_record_button.config(state=tk.DISABLED if active else tk.NORMAL)
        stop_record_button.config(state=tk.NORMAL if active else tk.DISABLED)
        export_button.config(state=tk.NORMAL if state["recorded_rows"] else tk.DISABLED)

    def _parameter_values():
        parameters = dict(mouse_predator=mouse_predator_var.get(), mouse_position=_mouse_position(), visual_range=visual_range_var.get(), wall_shapes=model.env.wall_shapes, seed=seed, mask=str(mask) if mask is not None else None,
                          desired_speed=speed_var.get(), fov_deg=fov_var.get(), repulsion_fov_deg=repulsion_fov_var.get(), spawn_radius=model.env.spawn_radius, capture_distance=capture_distance_var.get(),
                          width=model.env.width, height=model.env.height, bounded=model.env.bounded,
                          wall_avoid_radius=model.env.avoid_radius, wall_strength=model.env.avoid_strength,
                          thigmotaxis=model.env.thigmotaxis, interaction_model=model_var.get(),
                          steps_per_frame=steps_per_frame.get(), group_distance_multiplier=grouping_var.get())
        for t in all_types:
            parameters.update({f"agents.{t.name}.{key}": value for key, value in
                               {**t.to_dict(), "n_default": n_vars[t.name].get()}.items()})
        return {key: json.dumps(value, sort_keys=True) for key, value in parameters.items()}

    def _record_traits():
        if not record_var.get():
            return
        for name, agents in model.groups.items():
            atype = model.agent_types[name]
            for a in agents:
                previous = recorded_traits.get(a.unique_id)
                if previous is not None and name not in traits_dirty:
                    continue
                values = {key: float(a.individual.get(key, getattr(atype, key))) for key in _TRAIT_LABELS}
                for key, value in values.items():
                    if previous is None or value != previous[key]:
                        trait_writer.writerow((state["step"], state["step"] + 1, state["samples"],
                                               "initial" if previous is None else "change",
                                               a.unique_id, name, key, value))
                recorded_traits[a.unique_id] = values
        traits_dirty.clear()

    def _parameter_event(event, parameter, old_value, new_value):
        parameter_writer.writerow((state["step"], state["step"] + 1, state["samples"], event,
                                   parameter, old_value, new_value))

    def _log_parameter_changes():
        if not record_var.get():
            return
        current = _parameter_values()
        initial = not previous_parameters
        for key in sorted(current.keys() | previous_parameters.keys()):
            old, new = previous_parameters.get(key, "null"), current.get(key, "null")
            if initial or old != new:
                _parameter_event("initial" if initial else "change", key, old, new)
        previous_parameters.clear()
        previous_parameters.update(current)

    def _start_recording():
        mouse_trajectory.reset()
        if record_var.get():
            return
        record_var.set(True)
        traits_dirty.update(model.groups)
        _parameter_event("recording_start", "recording", "false", "true")
        _record_state()
        _refresh_info(force=True)

    def _stop_recording():
        if record_var.get():
            _log_parameter_changes()
            _record_traits()
            _parameter_event("recording_stop", "recording", "true", "false")
        record_var.set(False)
        _refresh_recording_controls()
        _refresh_info(force=True)

    def _store_simulation():
        if not state["recorded_rows"]:
            return
        was_running = state["active"]
        stop_sim()
        try:
            filename = filedialog.asksaveasfilename(parent=controls, title="Export agent trajectories",
                                                   defaultextension=".csv", initialfile="chimesim-trajectories.csv",
                                                   filetypes=[("CSV trajectories", "*.csv")])
            if not filename:
                return
            _log_parameter_changes()
            _record_traits()
            recording.flush()
            recording.seek(0)
            try:
                with open(filename, "w", newline="", encoding="utf-8") as output:
                    shutil.copyfileobj(recording, output)
            finally:
                recording.seek(0, 2)
            mouse_recording.flush()
            mouse_recording.seek(0)
            mouse_path = Path(filename).with_name(Path(filename).stem + '-mouse.csv')
            try:
                with mouse_path.open('w', newline='', encoding='utf-8') as output:
                    shutil.copyfileobj(mouse_recording, output)
            finally:
                mouse_recording.seek(0, 2)
            parameter_recording.flush()
            parameter_recording.seek(0)
            parameter_path = Path(filename).with_name(Path(filename).stem + "-parameters.csv")
            try:
                with parameter_path.open("w", newline="", encoding="utf-8") as output:
                    shutil.copyfileobj(parameter_recording, output)
            finally:
                parameter_recording.seek(0, 2)
            trait_recording.flush()
            trait_recording.seek(0)
            trait_path = Path(filename).with_name(Path(filename).stem + "-traits.csv")
            try:
                with trait_path.open("w", newline="", encoding="utf-8") as output:
                    shutil.copyfileobj(trait_recording, output)
            finally:
                trait_recording.seek(0, 2)
            status.set(f"Exported {state['recorded_rows']:,} trajectory rows")
        except OSError as exc:
            messagebox.showerror("Cannot export trajectories", str(exc), parent=controls)
        finally:
            if was_running:
                start_sim()
            _refresh_info(force=True)

    def _save_configuration():
        was_running = state["active"]
        stop_sim()
        try:
            filename = filedialog.asksaveasfilename(parent=controls, title="Save simulation settings",
                                                   defaultextension=".json", initialfile="chimesim-settings.json",
                                                   filetypes=[("JSON settings", "*.json")])
            if filename:
                with open(filename, "w", encoding="utf-8") as output:
                    json.dump(scenario_from_flat(_collect_settings()), output, indent=2)
        except OSError as exc:
            messagebox.showerror("Cannot save settings", str(exc), parent=controls)
        finally:
            if was_running:
                start_sim()

    def _load_configuration():
        was_running = state["active"]
        stop_sim()
        try:
            filename = filedialog.askopenfilename(parent=controls, title="Load simulation settings",
                                                  filetypes=[("JSON settings", "*.json"), ("All files", "*")])
            if not filename:
                return
            loaded = flatten(_load_file(filename))
            for old, new in (("speed", "desired_speed"), ("fov", "fov_deg")):
                if old in loaded:
                    loaded.setdefault(new, loaded.pop(old))
            window_layout = loaded.get("window_layout", {})
            if not isinstance(window_layout, dict) or any(not isinstance(v, dict) for v in window_layout.values()):
                raise ValueError("window_layout must map window names to layout objects")
            for key in ("bounded", "show_edges", "show_trails", "lock_arena_size"):
                if key in loaded and not isinstance(loaded[key], bool):
                    raise ValueError(f"{key} must be a boolean")
            for key in ("window_width", "window_height"):
                value = loaded.get(key, 400)
                if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                    raise ValueError(f"{key} must be an integer >= 2")
            for key, default in (("desired_speed", SPEED_LIMIT), ("fov_deg", FOV),
                                 ("capture_distance", CAPTURE_DISTANCE), ("trail_length", 50),
                                 ("group_distance_multiplier", 2.5), ("wall_avoid_radius", 50.), ("wall_strength", 6.), ("thigmotaxis", 0.)):
                value = loaded.get(key, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and nonnegative")
            if not 0 <= loaded.get("fov_deg", FOV) <= 360:
                raise ValueError("fov_deg must be between 0 and 360")
            if loaded.get("group_distance_multiplier", 2.5) <= 0:
                raise ValueError("group_distance_multiplier must be positive")
            selected = loaded.get("interaction_model", "zonal")
            if selected not in {"zonal", "voronoi"}:
                raise ValueError("interaction_model must be 'zonal' or 'voronoi'")
            repulsion_fov = loaded.get("repulsion_fov_deg")
            if repulsion_fov is not None and (not math.isfinite(repulsion_fov) or not 0 <= repulsion_fov <= 360):
                raise ValueError("repulsion_fov_deg must be between 0 and 360")
            _prepare_model(loaded)
            if state["recorded_rows"] and not messagebox.askokcancel(
                    "Load settings", "Loading starts a new simulation and clears the current recording.\n"
                    "Cancel to export your data first.", parent=controls):
                return
            state["reload_settings"] = loaded
            exit_sim()
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            messagebox.showerror("Cannot load settings", str(exc), parent=controls)
        finally:
            if was_running and not state["closing"]:
                start_sim()

    def _section(title):
        heading = tk.Frame(top)
        heading.pack(fill=tk.X, pady=(10, 3))
        tk.Frame(heading, height=1, bg="#aaaaaa").pack(fill=tk.X, pady=(0, 5))
        tk.Label(heading, text=title, font=("TkDefaultFont", 14, "bold"), anchor="w").pack(fill=tk.X)
        body = tk.Frame(top)
        body.pack(fill=tk.X)
        return body

    session = _section("Simulation")
    buttons = tk.Frame(session)
    buttons.pack(fill=tk.X)
    for label, command in (("Start", start_sim), ("Pause", stop_sim), ("Reset", reset_arena), ("Exit", exit_sim)):
        tk.Button(buttons, text=label, command=command).pack(side=tk.LEFT, padx=2)
    step_row = tk.Frame(session)
    step_row.pack(fill=tk.X)
    tk.Label(step_row, text="Steps per frame").pack(side=tk.LEFT)
    choice(step_row, steps_per_frame, 1, 5, 20, 100, command=lambda _: _log_parameter_changes()).pack(side=tk.LEFT)
    tk.Checkbutton(session, text="Mouse predator", variable=mouse_predator_var,
                   command=_toggle_mouse_predator).pack(anchor='w')
    arena = _section("Arena")
    size_row = tk.Frame(arena)
    size_row.pack(fill=tk.X)
    for label, var in (("Width", width_var), ("Height", height_var)):
        tk.Label(size_row, text=label).pack(side=tk.LEFT)
        tk.Entry(size_row, textvariable=var, width=7).pack(side=tk.LEFT, padx=3)
    tk.Button(arena, text="Apply size & reset", command=_apply_arena_size).pack(anchor="w", pady=4)
    tk.Checkbutton(arena, text="Lock arena size when resizing", variable=lock_var).pack(anchor="w")
    tk.Checkbutton(arena, text="Bounded arena", variable=bounded_var,
                   command=_on_bounded).pack(anchor="w")
    def _commit_environment(environment, new_mask):
        nonlocal mask
        stop_sim()
        model.env = environment
        mask = new_mask
        agents_now = model.all_agents
        points = np.array([[a.x, a.y] for a in agents_now], dtype=float).reshape(-1, 2)
        points %= [environment.width, environment.height]
        environment.project_free(points)
        for agent, (x, y) in zip(agents_now, points):
            agent.x, agent.y = float(x), float(y)
        width_var.set(str(environment.width))
        height_var.set(str(environment.height))
        trails.clear()
        positions.clear()
        position_colours.clear()
        info_history.clear()
        _wall_availability()
        mask_label.set(Path(mask).name if mask else "No image mask")
        _log_parameter_changes()
        _redraw_arena()
        draw()
        _refresh_info(force=True)

    def _change_walls(path, shapes, dimensions=None):
        try:
            w, h = dimensions or (model.env.width, model.env.height)
            environment = Environment(w, h, bounded=model.env.bounded, mask=path,
                                      wall_shapes=shapes, avoid_radius=wall_ar_var.get(),
                                      avoid_strength=wall_as_var.get(), thigmotaxis=wall_thig_var.get(), spawn_radius=model.env.spawn_radius)
            _commit_environment(environment, path)
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            messagebox.showerror("Cannot change walls", str(exc), parent=controls)

    def _load_wall_mask():
        stop_sim()
        path = filedialog.askopenfilename(parent=controls, title="Load wall mask",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*")])
        if not path:
            return
        try:
            from PIL import Image
            with Image.open(path) as image:
                dimensions = image.size
        except (OSError, ImportError) as exc:
            messagebox.showerror("Cannot load wall mask", str(exc), parent=controls)
            return
        resize = messagebox.askyesnocancel("Mask dimensions",
            f"Resize the arena to the image dimensions ({dimensions[0]} × {dimensions[1]})?\n"
            "Yes: use image dimensions. No: stretch the mask to the current arena.", parent=controls)
        if resize is not None:
            _change_walls(path, model.env.wall_shapes, dimensions if resize else None)

    mask_label = tk.StringVar(value=Path(mask).name if mask else "No image mask")
    tk.Label(arena, textvariable=mask_label, fg="#666666").pack(anchor="w")
    mask_buttons = tk.Frame(arena)
    mask_buttons.pack(fill=tk.X)
    tk.Button(mask_buttons, text="Load mask", command=_load_wall_mask).pack(side=tk.LEFT)
    tk.Button(mask_buttons, text="Remove mask", command=lambda: _change_walls(None, model.env.wall_shapes)).pack(side=tk.LEFT)
    wall_editor = WallEditor(arena, canvas, lambda: tuple(view), lambda: model.env,
                             stop_sim, lambda shapes: _change_walls(mask, shapes))
    def _camera_redraw():
        _redraw_arena()
        draw()

    def _zoom(factor):
        camera["zoom"] = min(16., max(.25, camera["zoom"] * factor))
        _camera_redraw()

    def _fit_arena():
        camera.update(zoom=1., x=.5, y=.5)
        _camera_redraw()

    def _scroll_camera(axis, *args):
        dimension = model.env.width if axis == "x" else model.env.height
        screen = canvas.winfo_width() if axis == "x" else canvas.winfo_height()
        fraction = screen / (dimension * view[0])
        if args[0] == "moveto":
            camera[axis] = float(args[1]) + fraction / 2
        else:
            camera[axis] += int(args[1]) * fraction * (.9 if args[2] == "pages" else .1)
        _camera_redraw()

    def _pan_start(event):
        state['mouse_pointer'] = (event.x, event.y)
        camera["drag"] = (event.x, event.y, camera["x"], camera["y"])

    def _pan_move(event):
        state['mouse_pointer'] = (event.x, event.y)
        if camera["drag"] is not None:
            x, y, cx, cy = camera["drag"]
            camera["x"] = cx - (event.x - x) / (model.env.width * view[0])
            camera["y"] = cy - (event.y - y) / (model.env.height * view[0])
            _camera_redraw()

    horizontal.config(command=lambda *args: _scroll_camera("x", *args))
    vertical.config(command=lambda *args: _scroll_camera("y", *args))
    bind_wheel(canvas, lambda event: _zoom(1.15 if wheel_units(event) < 0 else 1 / 1.15)
               if wheel_units(event) else None)
    for button in (2, 3):
        canvas.bind(f"<ButtonPress-{button}>", _pan_start)
        canvas.bind(f"<B{button}-Motion>", _pan_move)
        canvas.bind(f"<ButtonRelease-{button}>", lambda event: camera.update(drag=None))
    canvas.bind('<Motion>', _mouse_move, add='+')
    canvas.bind('<Enter>', _mouse_move, add='+')
    canvas.bind('<Leave>', _mouse_leave, add='+')
    wall_editor.mode.trace_add('write', lambda *_: _draw_mouse_predator())
    agents = _section("Agents and interaction model")
    model_row = tk.Frame(agents)
    model_row.pack(fill=tk.X)
    tk.Label(model_row, text="Model:").pack(side=tk.LEFT)
    choice(model_row, model_var, "zonal", "voronoi", command=_on_model).pack(side=tk.LEFT)
    row = tk.Frame(agents)
    row.pack(fill=tk.X)
    tk.Label(row, text="Agent types:").pack(side=tk.LEFT)
    for name in ("A1", "A2", "A3"):
        type_vars[name] = tk.BooleanVar(value=name in model.groups)
        type_checks[name] = tk.Checkbutton(row, text=name, variable=type_vars[name],
                                          command=lambda name=name: _toggle_type(name))
        type_checks[name].pack(side=tk.LEFT)
    population_frame = tk.Frame(agents)
    population_frame.pack(fill=tk.X)
    display = _section("Display")
    tk.Checkbutton(display, text="Show geometric (Voronoi) neighbours", variable=show_edges_var,
                   command=draw).pack(anchor="w")
    tk.Checkbutton(display, text="Show visual neighbours (directed)", variable=show_visual_var,
                   command=draw).pack(anchor="w")
    _slider_row(display, "Minimum visible fraction", visual_fraction_var, 0, 1, .05, lambda _: draw())
    tk.Checkbutton(display, text="Show information", variable=show_info_var,
                   command=lambda: _show(info) if show_info_var.get() else _hide(info, "information")).pack(anchor="w")
    tk.Checkbutton(display, text="Show scale bar and FPS", variable=show_scale_var, command=draw).pack(anchor="w")
    tk.Checkbutton(display, text="Show trajectory trails", variable=show_trails_var,
                   command=_toggle_trails).pack(anchor="w")
    tk.Checkbutton(display, text="Show individual positions", variable=show_positions_var,
                   command=lambda: _show_positions() if show_positions_var.get() else _hide(positions.window, "positions")).pack(anchor="w")
    tk.Checkbutton(display, text="Show interaction zones", variable=show_zones_var,
                   command=_toggle_zones).pack(anchor="w")
    camera_row = tk.Frame(display)
    camera_row.pack(fill=tk.X)
    tk.Label(camera_row, text="View:").pack(side=tk.LEFT)
    for label, callback in (("−", lambda: _zoom(1 / 1.25)), ("+", lambda: _zoom(1.25)), ("Fit arena", _fit_arena)):
        tk.Button(camera_row, text=label, command=callback).pack(side=tk.LEFT)
    style_row = tk.Frame(display)
    style_row.pack(fill=tk.X)
    tk.Label(style_row, text="Agent shape:").pack(side=tk.LEFT)
    choice(style_row, body_style, "Triangle", "Circle with tail", command=lambda _: draw()).pack(side=tk.LEFT)
    _slider_row(display, "Triangle size / tail length ×", display_size_var, .1, 5., .1, lambda _: draw())
    colour_row = tk.Frame(display)
    colour_row.pack(fill=tk.X)
    tk.Label(colour_row, text="Colour agents by:").pack(side=tk.LEFT)
    choice(colour_row, positions.colour_var, "Traits / types", "Spatial leadership", "Centre distance",
                  command=_change_position_colour).pack(side=tk.LEFT)
    _slider_row(display, "Trail length", trail_length_var, 1, 500, 1, _set_trail_length)
    _slider_row(display, "Group distance × NND", grouping_var, .5, max(10., grouping_var.get()), .25,
                _set_grouping_multiplier)
    storage = _section("Data and settings")
    record_buttons = tk.Frame(storage)
    record_buttons.pack(fill=tk.X)
    start_record_button = tk.Button(record_buttons, text="Start recording", command=_start_recording)
    start_record_button.pack(side=tk.LEFT)
    stop_record_button = tk.Button(record_buttons, text="Stop recording", command=_stop_recording)
    stop_record_button.pack(side=tk.LEFT)
    tk.Label(storage, textvariable=recording_status, anchor="w", fg="#777777",
             font=("TkDefaultFont", 10, "italic")).pack(fill=tk.X)
    export_button = tk.Button(storage, text="Export recorded data", command=_store_simulation)
    export_button.pack(anchor="w", pady=3)
    settings_buttons = tk.Frame(storage)
    settings_buttons.pack(fill=tk.X)
    tk.Button(settings_buttons, text="Save settings", command=_save_configuration).pack(side=tk.LEFT)
    tk.Button(settings_buttons, text="Load settings", command=_load_configuration).pack(side=tk.LEFT)

    _slider_row(parameter_top, "Speed (model pixels / step)", speed_var, 0, max(20, speed_var.get()), .5)
    _slider_row(parameter_top, "Field of view (°)", fov_var, 0, 360, 10, lambda _: draw())
    _slider_row(parameter_top, "Repulsion field of view (°)", repulsion_fov_var, 0, 360, 10, lambda _: draw())
    _slider_row(parameter_top, "Visual range", visual_range_var, 0, max(1000, visual_range_var.get()), 10, lambda _: draw())
    _slider_row(parameter_top, "Capture distance", capture_distance_var, 0, max(100, capture_distance_var.get()), 5)
    wall_options = tk.Frame(parameter_top)
    wall_options.pack(fill=tk.X)
    _slider_row(wall_options, "Wall avoid radius", wall_ar_var, 0, max(150, wall_ar_var.get()), 5,
                lambda value: setattr(model.env, "avoid_radius", float(value)))
    _slider_row(wall_options, "Wall strength", wall_as_var, 0, max(20, wall_as_var.get()), .5,
                lambda value: setattr(model.env, "avoid_strength", float(value)))
    follow_options = tk.Frame(wall_options)
    follow_options.pack(fill=tk.X)
    _slider_row(follow_options, "Wall follow", wall_thig_var, 0, max(3, wall_thig_var.get()), .05,
                lambda value: setattr(model.env, "thigmotaxis", float(value)))

    def _wall_availability():
        enabled = model.env.bounded or model.env.has_mask
        for widget in _descendants(wall_options):
            if isinstance(widget, (tk.Scale, tk.Button)):
                widget.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from _descendants(child)

    _wall_availability()
    menu = tk.Menu(root)
    simulation_menu = tk.Menu(menu, tearoff=False)
    for label, command in (("Start", start_sim), ("Pause", stop_sim), ("Reset", reset_arena), ("Exit", exit_sim)):
        simulation_menu.add_command(label=label, command=command)
    menu.add_cascade(label="Simulation", menu=simulation_menu)
    windows_menu = tk.Menu(menu, tearoff=False)
    menu.add_cascade(label="Windows", menu=windows_menu)
    root.config(menu=menu)
    _IMODEL_LABELS = {
        "zonal": ("ZOR", "ZOO", "ZOA"),
        "voronoi": ("Repulsion threshold", "Orientation distance", "Max interaction distance"),
    }
    _IMODEL_OPTIONS = ["zonal", "voronoi"]
    _rebuild_panels()
    root.update_idletasks()
    initial_size = (canvas.winfo_width(), canvas.winfo_height())
    state["canvas_size"] = initial_size if min(initial_size) > 1 else None
    _reset_recording()
    _redraw_arena()
    draw()
    _refresh_info(force=True)
    for window in (root, controls, parameters, info, positions.window):
        window.bind("<Configure>", _on_window_change)
        window.bind("<ButtonRelease-1>", _on_release)
    canvas.bind("<Configure>", _on_window_change)
    root.protocol("WM_DELETE_WINDOW", exit_sim)
    root.after(200, _raise_windows)
    try:
        root.mainloop()
    finally:
        # Close every resource even if another cleanup raises; destroy the
        # windows last and flush their native events before returning to Python.
        with ExitStack() as cleanup:
            cleanup.callback(destroy_window, root)
            cleanup.callback(mouse_recording.close)
            cleanup.callback(trait_recording.close)
            cleanup.callback(parameter_recording.close)
            cleanup.callback(recording.close)
            cleanup.callback(visual_network.close)
    return rows, state["reload_settings"]
