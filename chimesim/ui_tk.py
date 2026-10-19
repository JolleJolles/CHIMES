"""Interactive windows for viewing a fixed arena and controlling its model."""

import tkinter as tk
from tkinter import messagebox
import math
import time
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError
from chimesim.model import Model
from chimesim.agents import AgentType, BOID
from chimesim.config import SPEED_LIMIT, CATCH_DIST, FOV
from chimesim import metrics
from chimesim.environment import Environment
from chimesim.simulate import _load_file

_SETTINGS_PATH = Path.home() / ".chimesim" / "settings.json"
_MAX_TYPES = 3


def _arena_transform(width, height, view_width, view_height):
    """Uniform scale and centred offsets; viewing never changes model geometry."""
    scale = min(max(1, view_width - 24) / width, max(1, view_height - 24) / height)
    return scale, (view_width - width * scale) / 2, (view_height - height * scale) / 2


def _load_settings():
    path = _SETTINGS_PATH
    if not path.exists():
        path = Path.home() / ".absim" / "settings.json"
    if not path.exists():
        return {}
    try:
        return _load_file(path)
    except (OSError, ValueError) as exc:
        warnings.warn(f"Could not load GUI settings: {exc}", RuntimeWarning)
        return {}


def _save_settings(data):
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_PATH, "w") as handle:
            json.dump(data, handle, indent=2)
    except OSError as exc:
        warnings.warn(f"Could not save GUI settings: {exc}", RuntimeWarning)


def run(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None,
        width=None, height=None, window_width=None, window_height=None):
    """Open simulation, controls, information and per-agent-type windows.

    width/height are fixed arena dimensions in model pixels, independent of the
    display. window_width/window_height set the initial visualisation size in
    screen pixels. Explicit arguments override dictionary/JSON settings. With
    settings=None, restore the last GUI session. Resizing only scales the view;
    Apply size & reset changes the arena. Closing the simulation exits; closing
    a panel hides it, and the Windows menu restores it. log=True returns metrics
    for each model step after closing. Seed controls initial randomisation.
    """
    if settings is None:
        saved = _load_settings()
    elif isinstance(settings, (str, Path)):
        saved = _load_file(settings)
    elif isinstance(settings, dict):
        saved = dict(settings)
    else:
        raise TypeError("settings must be a dictionary or JSON path")
    if agent_types is not None:
        all_types = [t.clone() for t in agent_types]
    elif saved.get("agents"):
        all_types = [AgentType.from_dict(name, values) for name, values in saved["agents"].items()]
    else:
        all_types = [BOID.clone()]
    if not 1 <= len(all_types) <= _MAX_TYPES or len({t.name for t in all_types}) != len(all_types):
        raise ValueError(f"The GUI supports 1–{_MAX_TYPES} uniquely named agent types")
    for agent_type in all_types:
        agent_type.validate()
    width = saved.get("width", 1920) if width is None else width
    height = saved.get("height", 1080) if height is None else height
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (width, height)):
        raise ValueError("Arena width and height must be integers >= 2")
    bounded = saved.get("bounded", True) if bounded is None else bounded
    mask = saved.get("mask") if mask is None else mask
    seed = saved.get("seed") if seed is None else seed
    model = Model(agent_types=all_types, seed=seed)
    model.env = Environment(width, height, bounded=bounded, mask=mask,
                            avoid_radius=saved.get("wall_avoid_radius", 50.0),
                            avoid_strength=saved.get("wall_strength", 6.0),
                            thigmotaxis=saved.get("thigmotaxis", 0.0))
    for t in all_types:
        model.init_agents(t.name, t.n_default)

    root = tk.Tk()
    root.title("CHIMESim — Simulation")
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    view_w = saved.get("window_width", min(width, int(screen_w * .6))) if window_width is None else window_width
    view_h = saved.get("window_height", min(height, int(screen_h * .7))) if window_height is None else window_height
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (view_w, view_h)):
        root.destroy()
        raise ValueError("Window width and height must be integers >= 2")
    root.geometry(f"{view_w}x{view_h}+20+40")
    root.minsize(200, 150)
    canvas = tk.Canvas(root, bg="#e8e8e8", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    controls = tk.Toplevel(root)
    controls.title("CHIMESim — Controls")
    controls.geometry(f"350x{min(760, screen_h - 100)}+{max(0, screen_w - 370)}+40")
    controls.protocol("WM_DELETE_WINDOW", controls.withdraw)
    info = tk.Toplevel(root)
    info.title("CHIMESim — Information")
    info.geometry(f"410x300+{max(0, screen_w - 440)}+{max(80, screen_h - 350)}")
    info.protocol("WM_DELETE_WINDOW", info.withdraw)
    info_scroll = tk.Scrollbar(info)
    info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    info_canvas = tk.Canvas(info, bg="white", highlightthickness=0, yscrollcommand=info_scroll.set)
    info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    info_scroll.config(command=info_canvas.yview)
    def _info_wheel(event):
        delta = -int(event.delta / 120) if abs(event.delta) >= 120 else -int(event.delta)
        info_canvas.yview_scroll(delta, "units")
    info.bind("<MouseWheel>", _info_wheel)
    info.bind("<Button-4>", lambda _: info_canvas.yview_scroll(-1, "units"))
    info.bind("<Button-5>", lambda _: info_canvas.yview_scroll(1, "units"))

    state = dict(active=False, moving=False, timer=None, settle=None, remaining=0,
                 step=0, fps=0., last_frame=time.perf_counter(), last_info=0., closing=False)
    rows = [] if log else None
    status = tk.StringVar(value="Paused")
    n_vars, panels, panel_counts = {}, {}, {}
    speed_var = tk.DoubleVar(value=saved.get("desired_speed", saved.get("speed", SPEED_LIMIT)))
    fov_var = tk.DoubleVar(value=saved.get("fov_deg", saved.get("fov", FOV)))
    catch_dist_var = tk.DoubleVar(value=saved.get("catch_dist", CATCH_DIST))
    bounded_var = tk.BooleanVar(value=bounded)
    wall_ar_var = tk.DoubleVar(value=model.env.avoid_radius)
    wall_as_var = tk.DoubleVar(value=model.env.avoid_strength)
    wall_thig_var = tk.DoubleVar(value=model.env.thigmotaxis)
    show_edges_var = tk.BooleanVar(value=saved.get("show_edges", False))
    steps_per_frame = tk.IntVar(value=1)
    width_var, height_var = tk.StringVar(value=str(width)), tk.StringVar(value=str(height))
    view = [1., 0., 0.]

    def _scrollable(window):
        outer = tk.Frame(window)
        outer.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(outer)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        area = tk.Canvas(outer, highlightthickness=0, yscrollcommand=scroll.set)
        area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=area.yview)
        frame = tk.Frame(area, padx=8, pady=6)
        item = area.create_window(0, 0, window=frame, anchor="nw")
        frame.bind("<Configure>", lambda _: area.config(scrollregion=area.bbox("all")))
        area.bind("<Configure>", lambda event: area.itemconfig(item, width=event.width))

        def wheel(event):
            if event.num in (4, 5):
                units = -1 if event.num == 4 else 1
            else:
                units = -int(event.delta / 120) if abs(event.delta) >= 120 else -int(event.delta)
            area.yview_scroll(units, "units")
        window.bind("<MouseWheel>", wheel)
        window.bind("<Button-4>", wheel)
        window.bind("<Button-5>", wheel)
        return frame

    top = _scrollable(controls)

    def _redraw_arena():
        view[:] = _arena_transform(model.env.width, model.env.height,
                                   canvas.winfo_width(), canvas.winfo_height())
        scale, ox, oy = view
        canvas.delete("arena")
        canvas.delete("walls")
        canvas.create_rectangle(ox, oy, ox + model.env.width * scale, oy + model.env.height * scale,
                                fill="white", outline="#444444", width=2,
                                dash=() if model.env.bounded else (4, 4), tags="arena")
        if model.env.has_mask:
            wall = model.env._wall
            gh, gw = wall.shape
            sx, sy = model.env.width * scale / gw, model.env.height * scale / gh
            diff = np.diff(np.pad(wall.astype(np.int8), ((0, 0), (1, 1))), axis=1)
            for y in range(gh):
                starts, ends = np.where(diff[y] == 1)[0], np.where(diff[y] == -1)[0]
                for x0, x1 in zip(starts, ends):
                    canvas.create_rectangle(ox + x0 * sx, oy + y * sy,
                                            ox + x1 * sx, oy + (y + 1) * sy,
                                            fill="black", outline="", tags="walls")
        canvas.tag_lower("walls")
        canvas.tag_lower("arena")

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
            canvas.create_line(ox + pos[i, 0] * scale, oy + pos[i, 1] * scale,
                               ox + pos[j, 0] * scale, oy + pos[j, 1] * scale,
                               fill="#bbbbbb", tags="edges")

    def draw():
        canvas.delete("agents")
        _draw_edges()
        scale, ox, oy = view
        for agent in model.all_agents:
            if not math.isfinite(agent.x) or not math.isfinite(agent.y):
                continue
            angle = math.atan2(agent.dy, agent.dx)
            ca, sa = math.cos(angle), math.sin(angle)
            x, y = ox + agent.x * scale, oy + agent.y * scale
            length, half_width = agent.type.size * scale, agent.type.size * scale / 3
            front, back = 2 * length / 3, length / 3
            canvas.create_polygon(x + front * ca, y + front * sa,
                                  x - back * ca + half_width * sa, y - back * sa - half_width * ca,
                                  x - back * ca - half_width * sa, y - back * sa + half_width * ca,
                                  fill=agent.color or agent.type.color, outline="", tags="agents")

    def _refresh_info(force=False):
        now = time.perf_counter()
        if not force and now - state["last_info"] < .15:
            return
        state["last_info"] = now
        result = metrics.compute(model.groups, model.env)
        info_canvas.delete("all")
        text = (f"{status.get()}    Step {state['step']}    {state['fps']:.0f} fps\n"
                f"Arena: {model.env.width} × {model.env.height} model pixels")
        info_canvas.create_text(12, 10, anchor="nw", text=text, font=("TkFixedFont", 11))
        y = 60
        for name, atype in model.agent_types.items():
            text = (f"{name}: N={result[f'{name}_n']}   "
                    f"Polarisation={result[f'{name}_polarisation']:.2f}   "
                    f"NND={result[f'{name}_nnd']:.1f}")
            info_canvas.create_text(12, y, anchor="nw", text=text, fill=atype.color)
            y += 25
            if atype.has_colorbar:
                label = getattr(atype, "_color_label", None) or atype.color_by
                info_canvas.create_text(12, y, anchor="nw", text=label.replace("_", " "))
                y += 20
                for i, colour in enumerate(atype._colorbar):
                    info_canvas.create_rectangle(12 + i * 5, y, 17 + i * 5, y + 10,
                                                 fill=colour, outline="")
                y += 14
                info_canvas.create_text(12, y, anchor="nw", text=f"{atype._dist_min:.3g}")
                info_canvas.create_text(252, y, anchor="ne", text=f"{atype._dist_max:.3g}")
                y += 25
        info_canvas.config(scrollregion=(0, 0, 410, y + 12))

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
        kw = dict(catch_dist=catch_dist_var.get(), desired_speed=speed_var.get(), fov_deg=fov_var.get())
        try:
            while state["remaining"] > 0:
                model.step(**kw)
                state["step"] += 1
                state["remaining"] -= 1
                if rows is not None:
                    rows.append({"step": state["step"], **metrics.compute(model.groups, model.env)})
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
        state["moving"] = False
        state["last_frame"] = time.perf_counter()
        status.set("Running" if state["active"] else "Paused")
        _redraw_arena()
        draw()
        _refresh_info(force=True)
        _schedule()

    def _on_window_change(event):
        if state["closing"] or event.widget not in (root, canvas, controls, info, *panels.values()):
            return
        state["moving"] = True
        _cancel_frame()
        status.set("Resizing / moving" if state["active"] else "Paused")
        if state["settle"] is not None:
            root.after_cancel(state["settle"])
        state["settle"] = root.after(180, _finish_move)

    def _on_release(_event):
        if state["moving"]:
            _finish_move()

    def _set_group_count(name, count):
        agents = model.groups[name]
        if count > len(agents):
            model.init_agents(name, count - len(agents))
        elif count < len(agents):
            keep = set(model.rng.choice(len(agents), size=count, replace=False).tolist())
            model.groups[name] = [agent for i, agent in enumerate(agents) if i in keep]
        draw()
        _refresh_info(force=True)

    def _nudge(var, delta, lo, hi, cb=None):
        v = max(lo, min(hi, var.get() + delta))
        var.set(v)
        if cb:
            cb(v)

    def _slider_row(parent, label, var, lo, hi, res, cb=None):
        # label may be a plain str or a tk.StringVar (for dynamic relabelling)
        if isinstance(label, tk.StringVar):
            tk.Label(parent, textvariable=label, anchor=tk.W).pack(fill=tk.X)
        else:
            tk.Label(parent, text=label, anchor=tk.W).pack(fill=tk.X)
        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        tk.Button(row, text="◀", width=2,
                  command=lambda: _nudge(var, -res, lo, hi, cb)).pack(side=tk.LEFT)
        kw = dict(from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL, variable=var)
        if cb:
            kw["command"] = cb
        tk.Scale(row, **kw).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="▶", width=2,
                  command=lambda: _nudge(var, +res, lo, hi, cb)).pack(side=tk.LEFT)
        tk.Frame(parent, height=1, bg="#cccccc").pack(fill=tk.X, pady=2)

    def _build_panel(atype):
        """Create an independently movable, scrollable agent-type window."""
        window = tk.Toplevel(root)
        window.title(f"CHIMESim — {atype.name}")
        index = all_types.index(atype)
        window.geometry(f"330x620+{min(80 + index * 340, max(0, screen_w - 350))}+100")
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        col = _scrollable(window)
        n_init = panel_counts.get(atype.name, len(model.groups[atype.name]))

        # ── Interaction model dropdown ──────────────────────────────────
        imodel_var = tk.StringVar(value=atype.interaction_model)
        zor_lbl = tk.StringVar(value=_IMODEL_LABELS[atype.interaction_model][0])
        zoo_lbl = tk.StringVar(value=_IMODEL_LABELS[atype.interaction_model][1])
        zoa_lbl = tk.StringVar(value=_IMODEL_LABELS[atype.interaction_model][2])

        def _on_imodel(val, t=atype, lz=zor_lbl, lo=zoo_lbl, la=zoa_lbl):
            t.interaction_model = val
            lbls = _IMODEL_LABELS[val]
            lz.set(lbls[0]);  lo.set(lbls[1]);  la.set(lbls[2])

        irow = tk.Frame(col)
        irow.pack(fill=tk.X, pady=(0, 4))
        tk.Label(irow, text="Model:", anchor=tk.W).pack(side=tk.LEFT)
        tk.OptionMenu(irow, imodel_var, *_IMODEL_OPTIONS,
                      command=_on_imodel).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(col, height=1, bg="#cccccc").pack(fill=tk.X, pady=2)

        # ── N agents ────────────────────────────────────────────────────
        n_var = tk.IntVar(value=n_init)
        n_vars[atype.name] = n_var
        def _on_n(val, t=atype): _set_group_count(t.name, int(float(val)))
        _slider_row(col, "N", n_var, 0, 500, 1, _on_n)

        # ── Zone / Voronoi sliders (labels swap with model) ─────────────
        zor_v = tk.DoubleVar(value=atype.r_r)
        def _on_zor(val, t=atype):
            t.r_r = float(val)
            t.r_o = max(t.r_o, t.r_r)
            t.r_a = max(t.r_a, t.r_o)
            zoo_v.set(t.r_o)
            zoa_v.set(t.r_a)
        _slider_row(col, zor_lbl, zor_v, 0, max(50, atype.r_r), 1, _on_zor)

        zoo_v = tk.DoubleVar(value=atype.r_o)
        def _on_zoo(val, t=atype):
            t.r_o = max(float(val), t.r_r)
            t.r_a = max(t.r_a, t.r_o)
            zoo_v.set(t.r_o)
            zoa_v.set(t.r_a)
        _slider_row(col, zoo_lbl, zoo_v, 0, max(150, atype.r_o), 5, _on_zoo)

        zoa_v = tk.DoubleVar(value=atype.r_a)
        def _on_zoa(val, t=atype):
            t.r_a = max(float(val), t.r_o)
            zoa_v.set(t.r_a)
        _slider_row(col, zoa_lbl, zoa_v, 0, max(300, atype.r_a), 5, _on_zoa)

        # ── Fixed sliders ────────────────────────────────────────────────
        det_v = tk.DoubleVar(value=atype.flee_dist)
        def _on_det(val, t=atype): t.flee_dist = float(val)
        _slider_row(col, "Detect range", det_v, 0, max(400, atype.flee_dist), 10, _on_det)

        ss_v = tk.DoubleVar(value=atype.speed_scale)
        def _on_ss(val, t=atype): t.speed_scale = float(val)
        _slider_row(col, "Speed scale", ss_v, 0, max(5.0, atype.speed_scale), 0.1, _on_ss)

        sv_v = tk.DoubleVar(value=atype.social_speed)
        def _on_sv(val, t=atype): t.social_speed = float(val)
        _slider_row(col, "Social speed", sv_v, 0.0, 1.0, 0.05, _on_sv)

        mt_v = tk.DoubleVar(value=round(math.degrees(atype.max_turn)))
        def _on_mt(val, t=atype): t.max_turn = math.radians(float(val))
        _slider_row(col, "Max turn (°)", mt_v, 0, 180, 1, _on_mt)

        no_v = tk.DoubleVar(value=round(math.degrees(atype.noise)))
        def _on_no(val, t=atype): t.noise = math.radians(float(val))
        _slider_row(col, "Noise σ (°)", no_v, 0, max(90, no_v.get()), 1, _on_no)

        # ── Predation checkboxes ─────────────────────────────────────────
        others = [t for t in all_types if t.name != atype.name]
        if others:
            tk.Frame(col, height=1, bg="#999999").pack(fill=tk.X, pady=4)
            for other in others:
                h_v = tk.BooleanVar(value=(other.name in atype.hunts))
                def _on_h(ot=other, at=atype, hv=h_v):
                    if hv.get(): at.hunts.add(ot.name)
                    else:        at.hunts.discard(ot.name)
                tk.Checkbutton(col, text=f"hunts  {other.name}",
                               variable=h_v, command=_on_h).pack(anchor=tk.W)
                f_v = tk.BooleanVar(value=(other.name in atype.flees))
                def _on_f(ot=other, at=atype, fv=f_v):
                    if fv.get(): at.flees.add(ot.name)
                    else:        at.flees.discard(ot.name)
                tk.Checkbutton(col, text=f"flees  {other.name}",
                               variable=f_v, command=_on_f).pack(anchor=tk.W)

        window.bind("<Configure>", _on_window_change)
        window.bind("<ButtonRelease-1>", _on_release)
        return window

    def _show(window):
        window.deiconify()
        window.lift()

    def _rebuild_panels():
        panel_counts.update({name: var.get() for name, var in n_vars.items()})
        for window in panels.values():
            window.destroy()
        panels.clear()
        n_vars.clear()
        for t in all_types:
            panels[t.name] = _build_panel(t)
        count_label.config(text=str(len(all_types)))
        windows_menu.delete(0, "end")
        windows_menu.add_command(label="Controls", command=lambda: _show(controls))
        windows_menu.add_command(label="Information", command=lambda: _show(info))
        windows_menu.add_separator()
        for name, window in panels.items():
            windows_menu.add_command(label=f"Agents: {name}", command=lambda w=window: _show(w))
        _refresh_info(force=True)

    def _add_type():
        if len(all_types) >= _MAX_TYPES:
            return
        index = 2
        while f"type_{index}" in model.groups:
            index += 1
        t = all_types[-1].clone(f"type_{index}")
        t.color = ("#fc9056", "#56aa9a")[(len(all_types) - 1) % 2]
        t.n_default = n_vars[all_types[-1].name].get()
        t.hunts.discard(t.name)
        t.flees.discard(t.name)
        all_types.append(t)
        model.agent_types[t.name] = t
        model.groups[t.name] = []
        model.init_agents(t.name, t.n_default)
        _rebuild_panels()
        draw()

    def _remove_type():
        if len(all_types) <= 1:
            return
        t = all_types.pop()
        del model.groups[t.name]
        del model.agent_types[t.name]
        for remaining in all_types:
            remaining.hunts.discard(t.name)
            remaining.flees.discard(t.name)
        _rebuild_panels()
        panel_counts.pop(t.name, None)
        draw()

    def _collect_settings():
        return dict(desired_speed=speed_var.get(), fov_deg=fov_var.get(), catch_dist=catch_dist_var.get(),
                    bounded=model.env.bounded, width=model.env.width, height=model.env.height,
                    window_width=canvas.winfo_width(), window_height=canvas.winfo_height(),
                    wall_avoid_radius=wall_ar_var.get(), wall_strength=wall_as_var.get(),
                    thigmotaxis=wall_thig_var.get(), show_edges=show_edges_var.get(),
                    mask=str(Path(mask).expanduser().resolve()) if mask is not None else None,
                    agents={t.name: {**t.to_dict(), "n_default": n_vars[t.name].get()} for t in all_types})

    def reset_arena():
        stop_sim()
        model.clear()
        state["step"] = 0
        if rows is not None:
            rows.clear()
        for t in all_types:
            model.init_agents(t.name, n_vars[t.name].get())
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
                                      thigmotaxis=wall_thig_var.get())
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Cannot change arena size", str(exc), parent=controls)
            return
        model.env = environment
        reset_arena()

    def _on_bounded():
        model.env.bounded = bounded_var.get()
        _redraw_arena()
        draw()
        _refresh_info(force=True)

    def exit_sim():
        state["closing"] = True
        _cancel_frame()
        if state["settle"] is not None:
            root.after_cancel(state["settle"])
        _save_settings(_collect_settings())
        root.destroy()

    buttons = tk.Frame(top)
    buttons.pack(fill=tk.X)
    for label, command in (("Start", start_sim), ("Pause", stop_sim), ("Reset", reset_arena), ("Exit", exit_sim)):
        tk.Button(buttons, text=label, command=command).pack(side=tk.LEFT, padx=2)
    tk.Label(top, textvariable=status, anchor="w").pack(fill=tk.X, pady=4)
    tk.Label(top, text="Steps per frame", anchor="w").pack(fill=tk.X)
    tk.OptionMenu(top, steps_per_frame, 1, 5, 20, 100).pack(anchor="w")
    tk.Label(top, text="Agent types", anchor="w").pack(fill=tk.X)
    row = tk.Frame(top)
    row.pack(fill=tk.X)
    tk.Button(row, text="Remove type", command=_remove_type).pack(side=tk.LEFT)
    count_label = tk.Label(row, text=str(len(all_types)), width=3)
    count_label.pack(side=tk.LEFT)
    tk.Button(row, text="Add type", command=_add_type).pack(side=tk.LEFT)

    tk.Label(top, text="Movement", font=("", 11, "bold"), anchor="w").pack(fill=tk.X, pady=(10, 0))
    _slider_row(top, "Speed (model pixels / step)", speed_var, 0, max(20, speed_var.get()), .5)
    _slider_row(top, "Field of view (°)", fov_var, 0, 360, 10)
    _slider_row(top, "Catch distance", catch_dist_var, 0, max(100, catch_dist_var.get()), 5)
    tk.Label(top, text="Arena size (model pixels)", font=("", 11, "bold"), anchor="w").pack(fill=tk.X, pady=(10, 0))
    size_row = tk.Frame(top)
    size_row.pack(fill=tk.X)
    for label, var in (("Width", width_var), ("Height", height_var)):
        tk.Label(size_row, text=label).pack(side=tk.LEFT)
        tk.Entry(size_row, textvariable=var, width=7).pack(side=tk.LEFT, padx=3)
    tk.Button(top, text="Apply size & reset", command=_apply_arena_size).pack(anchor="w", pady=4)
    tk.Label(top, text="Resizing a window only changes the view.", anchor="w").pack(fill=tk.X)
    tk.Checkbutton(top, text="Bounded arena (outer walls)", variable=bounded_var,
                   command=_on_bounded).pack(anchor="w")
    _slider_row(top, "Wall avoid radius", wall_ar_var, 0, max(150, wall_ar_var.get()), 5,
                lambda value: setattr(model.env, "avoid_radius", float(value)))
    _slider_row(top, "Wall strength", wall_as_var, 0, max(20, wall_as_var.get()), .5,
                lambda value: setattr(model.env, "avoid_strength", float(value)))
    _slider_row(top, "Wall follow", wall_thig_var, 0, max(3, wall_thig_var.get()), .05,
                lambda value: setattr(model.env, "thigmotaxis", float(value)))
    tk.Checkbutton(top, text="Show geometric neighbours", variable=show_edges_var,
                   command=draw).pack(anchor="w")
    tk.Button(top, text="Show information", command=lambda: _show(info)).pack(anchor="w")

    menu = tk.Menu(root)
    simulation_menu = tk.Menu(menu, tearoff=False)
    for label, command in (("Start", start_sim), ("Pause", stop_sim), ("Reset", reset_arena), ("Exit", exit_sim)):
        simulation_menu.add_command(label=label, command=command)
    menu.add_cascade(label="Simulation", menu=simulation_menu)
    windows_menu = tk.Menu(menu, tearoff=False)
    menu.add_cascade(label="Windows", menu=windows_menu)
    root.config(menu=menu)
    _IMODEL_LABELS = {
        "zonal": ("ZOR (repulsion radius)", "ZOO (orientation radius)", "ZOA (attraction radius)"),
        "voronoi": ("Repulsion threshold", "Orientation distance", "Max interaction distance"),
    }
    _IMODEL_OPTIONS = ["zonal", "voronoi"]
    _rebuild_panels()
    root.update_idletasks()
    _redraw_arena()
    draw()
    _refresh_info(force=True)
    for window in (root, controls, info):
        window.bind("<Configure>", _on_window_change)
        window.bind("<ButtonRelease-1>", _on_release)
    canvas.bind("<Configure>", _on_window_change)
    root.protocol("WM_DELETE_WINDOW", exit_sim)
    root.mainloop()
    return rows
