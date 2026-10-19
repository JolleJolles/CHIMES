import tkinter as tk
import math
import random
import time
import json
import warnings
import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay, QhullError
from ABSim.model import Model
from ABSim.agents import AgentType, BOID
import ABSim.config as config
from ABSim.config import SPEED_LIMIT, CATCH_DIST, FOV
import ABSim.metrics as metrics
from ABSim.environment import Environment
from ABSim.simulate import _load_file

_SETTINGS_PATH = Path.home() / ".absim" / "settings.json"

# Pre-defined extra type slots (added when the user increases the type count)
_EXTRA_SLOTS = [
    dict(name="type_2", color="#fc9056", size=14, n_default=20),
    dict(name="type_3", color="#56fc9a", size=14, n_default=20),
]
_MAX_TYPES = 1 + len(_EXTRA_SLOTS)


def _load_settings():
    if not _SETTINGS_PATH.exists():
        return {}
    try:
        return _load_file(_SETTINGS_PATH)
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


def run(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None):
    """
    Launch the interactive simulation window.

    Parameters
    ----------
    agent_types : list of AgentType, or None for default single BOID group
    log         : if True, collect metrics every model step and return on close
    mask        : path to a wall mask image (dark = wall, white = free)
    bounded     : override the saved bounded-arena setting
    settings    : dictionary or JSON file; None restores GUI auto-save
    seed        : optional initial random seed
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
    if bounded is None:
        bounded = saved.get("bounded", True)
    if mask is None:
        mask = saved.get("mask")

    root = tk.Tk()
    root.title("ABSim")
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

    paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=6,
                           sashrelief=tk.RAISED, bg="#aaaaaa")
    paned.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(paned, bg="white")
    paned.add(canvas, stretch="always", minsize=400)

    # Pre-allocate Delaunay edge pool — updated with coords() each frame,
    # which is much faster than create/delete.  3000 covers ~1000 agents.
    _EDGE_POOL = 3000
    _edge_pool = [
        canvas.create_line(0, 0, 1, 1, fill="#aaaaaa", width=1,
                           state="hidden", tags="edges")
        for _ in range(_EDGE_POOL)
    ]
    _n_edges_prev = [0]   # how many pool items were visible last frame

    def _on_canvas_resize(event):
        new_w, new_h = event.width, event.height
        if new_w <= 1 or new_h <= 1:
            return                            # ignore spurious events during layout
        old_w, old_h = config.WIDTH, config.HEIGHT
        config.WIDTH  = new_w
        config.HEIGHT = new_h
        # Rescale agent positions proportionally so they don't pile up at edges
        if old_w > 1 and old_h > 1 and (old_w != new_w or old_h != new_h):
            sx, sy = new_w / old_w, new_h / old_h
            try:
                for agents in model.groups.values():
                    for a in agents:
                        a.x = max(0.0, min(a.x * sx, new_w - 1.0))
                        a.y = max(0.0, min(a.y * sy, new_h - 1.0))
            except (NameError, AttributeError):
                pass
        if _env is not None:
            _env.resize(new_w, new_h)
            if _env.has_mask:
                _show_mask_overlay()
    canvas.bind("<Configure>", _on_canvas_resize)

    # ================================================================== #
    # Right panel: fixed top strip + scrollable agent columns below
    # ================================================================== #
    right = tk.Frame(paned, bd=1, relief=tk.FLAT)
    paned.add(right, stretch="never", minsize=200)

    top = tk.Frame(right, padx=6, pady=4)
    top.pack(side=tk.TOP, fill=tk.X)

    scroll_outer = tk.Frame(right)
    scroll_outer.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    vbar = tk.Scrollbar(scroll_outer, orient=tk.VERTICAL)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)

    agent_cv = tk.Canvas(scroll_outer, yscrollcommand=vbar.set, highlightthickness=0)
    agent_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vbar.config(command=agent_cv.yview)

    inner  = tk.Frame(agent_cv)
    win_id = agent_cv.create_window((0, 0), window=inner, anchor="nw")

    inner.bind("<Configure>",
               lambda _: agent_cv.configure(scrollregion=agent_cv.bbox("all")))
    agent_cv.bind("<Configure>",
                  lambda e: agent_cv.itemconfig(win_id, width=e.width))

    def _wheel(event):
        agent_cv.yview_scroll(int(-1 * (event.delta / 120)), "units")
    agent_cv.bind_all("<MouseWheel>", _wheel)

    # ================================================================== #
    # Simulation state
    # ================================================================== #
    running   = {"id": None}
    _fps      = [time.perf_counter(), 0.0]
    _log      = [] if log else None
    _step     = [0]
    _HUD_FONT = ("TkFixedFont", 13)

    def draw():
        canvas.delete("boundary")
        canvas.delete("agents")
        if _env is not None and _env.bounded:
            canvas.create_rectangle(
                1, 1, config.WIDTH - 2, config.HEIGHT - 2,
                outline="#444444", width=3, tags="boundary")
        _check_walls = _env is not None and _env.has_mask
        for agents in model.groups.values():
            for a in agents:
                if math.isnan(a.x) or math.isnan(a.y):
                    continue
                if _check_walls:
                    xi = max(0, min(int(a.x * _env._gw / _env.width), _env._gw - 1))
                    yi = max(0, min(int(a.y * _env._gh / _env.height), _env._gh - 1))
                    if _env._wall[yi, xi]:
                        continue
                angle  = math.atan2(a.dy, a.dx)
                x, y   = a.x, a.y
                L, w   = a.type.size, a.type.size / 3
                tip_f, base_b = 2 * L / 3, L / 3
                ca, sa = math.cos(angle), math.sin(angle)
                p1 = (x + tip_f  * ca,           y + tip_f  * sa)
                p2 = (x - base_b * ca + w * sa,  y - base_b * sa - w * ca)
                p3 = (x - base_b * ca - w * sa,  y - base_b * sa + w * ca)
                canvas.create_polygon([p1, p2, p3],
                                      fill=a.color or a.type.color,
                                      outline="", tags="agents")

    def _show_mask_overlay():
        canvas.delete("walls")
        if _env is None or _env._wall is None:
            return
        wall = _env._wall
        gh, gw = wall.shape
        sx = config.WIDTH  / gw
        sy = config.HEIGHT / gh
        padded = np.pad(wall.astype(np.int8), ((0, 0), (1, 1)), constant_values=0)
        diff = np.diff(padded, axis=1)
        for y in range(gh):
            starts = np.where(diff[y] ==  1)[0]
            ends   = np.where(diff[y] == -1)[0]
            y0, y1 = int(y * sy), int((y + 1) * sy)
            for x0, x1 in zip(starts.tolist(), ends.tolist()):
                canvas.create_rectangle(
                    int(x0 * sx), y0, int(x1 * sx), y1,
                    fill="black", outline="", tags="walls"
                )
        canvas.tag_lower("walls")

    def _draw_hud(m):
        canvas.delete("hud")

        # ── left: fps + per-group stats ─────────────────────────────────
        y = 8
        canvas.create_text(8, y, anchor=tk.NW, text=f"{_fps[1]:.0f} fps",
                           fill="#333333", font=_HUD_FONT, tags="hud")
        y += 20
        for name, atype in model.agent_types.items():
            n   = m.get(f"{name}_n", 0)
            p   = m.get(f"{name}_polarisation", 0.0)
            nnd = m.get(f"{name}_nnd", 0.0)
            canvas.create_text(
                8, y, anchor=tk.NW,
                text=f"{name:<12s}  N={n:>3d}  P={p:.2f}  NND={nnd:>5.1f}",
                fill=atype.color, font=_HUD_FONT, tags="hud")
            y += 20

        # ── right: one colorbar per agent type that has color_by set ────
        BAR_W, BAR_H = 160, 11
        bx = config.WIDTH - BAR_W - 14
        by = 8
        for name, atype in model.agent_types.items():
            if not atype.has_colorbar:
                continue
            param_label = getattr(atype, "_color_label", atype.color_by).replace("_", " ")
            # Title: "group · param"
            canvas.create_text(bx, by, anchor=tk.NW,
                               text=f"{name} · {param_label}",
                               fill="#222222", font=_HUD_FONT, tags="hud")
            by += 15
            # Gradient bar (48 pre-computed colour cells)
            n_cells = len(atype._colorbar)
            cw = BAR_W / n_cells
            for i, hex_c in enumerate(atype._colorbar):
                canvas.create_rectangle(
                    int(bx + i * cw), by,
                    int(bx + (i + 1) * cw), by + BAR_H,
                    fill=hex_c, outline="", tags="hud")
            # Min / max value labels
            canvas.create_text(bx,          by + BAR_H + 2, anchor=tk.NW,
                               text=f"{atype._dist_min:.3g}",
                               fill="#222222", font=_HUD_FONT, tags="hud")
            canvas.create_text(bx + BAR_W,  by + BAR_H + 2, anchor=tk.NE,
                               text=f"{atype._dist_max:.3g}",
                               fill="#222222", font=_HUD_FONT, tags="hud")
            by += BAR_H + 22

    def _draw_edges():
        """Draw Delaunay neighbour edges for all agents using the pre-allocated pool."""
        # Collect valid agent positions (skip NaN / in-wall agents)
        all_pos, all_colors = [], []
        _check_walls = _env is not None and _env.has_mask
        for agents in model.groups.values():
            for a in agents:
                if math.isnan(a.x) or math.isnan(a.y):
                    continue
                if _check_walls:
                    xi = max(0, min(int(a.x * _env._gw / _env.width), _env._gw - 1))
                    yi = max(0, min(int(a.y * _env._gh / _env.height), _env._gh - 1))
                    if _env._wall[yi, xi]:
                        continue
                all_pos.append((a.x, a.y))
                all_colors.append(a.color or a.type.color)

        n = len(all_pos)
        n_prev = _n_edges_prev[0]

        if n < 2:
            # Not enough points — hide whatever was shown before
            for k in range(n_prev):
                canvas.itemconfig(_edge_pool[k], state="hidden")
            _n_edges_prev[0] = 0
            return

        pos = np.array(all_pos)
        try:
            tri = Delaunay(pos) if n >= 3 else None
        except QhullError:
            for k in range(n_prev):
                canvas.itemconfig(_edge_pool[k], state="hidden")
            _n_edges_prev[0] = 0
            return

        # Collect unique edges from Delaunay simplices
        seen = set()
        edges = [(0, 1)] if n == 2 else []
        for simplex in (tri.simplices if tri is not None else []):
            for a in range(3):
                i, j = simplex[a], simplex[(a + 1) % 3]
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    edges.append(key)

        n_edges = min(len(edges), _EDGE_POOL)

        # Update visible pool items
        for k, (i, j) in enumerate(edges[:n_edges]):
            canvas.coords(_edge_pool[k],
                          pos[i, 0], pos[i, 1], pos[j, 0], pos[j, 1])
            if k >= n_prev:   # was hidden — need to show
                canvas.itemconfig(_edge_pool[k], state="normal")

        # Hide pool items no longer needed
        for k in range(n_edges, n_prev):
            canvas.itemconfig(_edge_pool[k], state="hidden")

        _n_edges_prev[0] = n_edges
        canvas.tag_lower("edges")   # keep edges behind agents + walls

    def loop():
        now = time.perf_counter()
        dt  = now - _fps[0]
        _fps[0] = now
        if dt > 0:
            _fps[1] = 0.8 * _fps[1] + 0.2 / dt

        kw = dict(catch_dist=catch_dist_var.get(),
                  desired_speed=speed_var.get(),
                  fov_deg=fov_var.get())
        for _ in range(_spf[0]):
            model.step(**kw)
            _step[0] += 1
            if _log is not None:
                _log.append({"step": _step[0], **metrics.compute(model.groups, model.env)})
        draw()
        if show_edges_var.get():
            _draw_edges()
        elif _n_edges_prev[0] > 0:
            # Toggle just turned off — clear remaining edges
            for k in range(_n_edges_prev[0]):
                canvas.itemconfig(_edge_pool[k], state="hidden")
            _n_edges_prev[0] = 0

        m = metrics.compute(model.groups, model.env)
        _draw_hud(m)


        root.update_idletasks()
        running["id"] = root.after(0, loop)

    def start_sim():
        if running["id"] is None:
            loop()

    def stop_sim():
        if running["id"] is not None:
            root.after_cancel(running["id"])
            running["id"] = None

    def _set_group_count(type_name, new_n):
        agents    = model.groups[type_name]
        current_n = len(agents)
        if new_n > current_n:
            model.init_agents(type_name, new_n - current_n)
        elif new_n < current_n:
            keep = set(random.sample(range(current_n), new_n))
            model.groups[type_name] = [a for i, a in enumerate(agents) if i in keep]

    # ================================================================== #
    # Widget helpers
    # ================================================================== #
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

    # ================================================================== #
    # Per-type panel builder + dynamic count
    # ================================================================== #
    n_vars:  dict[str, tk.IntVar]  = {}
    _panels: dict[str, tk.Widget]  = {}

    # Slider labels that differ between interaction models
    _IMODEL_LABELS = {
        "zonal":   ("ZOR (repulsion radius)",  "ZOO (orientation radius)", "ZOA (attraction radius)"),
        "voronoi": ("Repulsion threshold",      "Orientation distance",     "Max interaction dist"),
    }
    _IMODEL_OPTIONS = ["zonal", "voronoi"]

    def _build_panel(atype):
        """Create and pack one agent-type column.  Returns the LabelFrame."""
        try:
            n_live = len(model.groups.get(atype.name, []))
            n_init = n_live if n_live > 0 else atype.n_default
        except NameError:
            n_init = atype.n_default

        col = tk.LabelFrame(
            inner,
            text=f"  {atype.name.capitalize()}  ",
            fg=atype.color, font=("", 10, "bold"),
            bd=2, relief=tk.GROOVE, padx=5, pady=5,
        )
        col.pack(side=tk.TOP, fill=tk.X, padx=4, pady=6)

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
        _slider_row(col, zor_lbl, zor_v, 1, 50, 1, _on_zor)

        zoo_v = tk.DoubleVar(value=atype.r_o)
        def _on_zoo(val, t=atype):
            t.r_o = max(float(val), t.r_r)
            t.r_a = max(t.r_a, t.r_o)
            zoo_v.set(t.r_o)
            zoa_v.set(t.r_a)
        _slider_row(col, zoo_lbl, zoo_v, 5, 150, 5, _on_zoo)

        zoa_v = tk.DoubleVar(value=atype.r_a)
        def _on_zoa(val, t=atype):
            t.r_a = max(float(val), t.r_o)
            zoa_v.set(t.r_a)
        _slider_row(col, zoa_lbl, zoa_v, 10, 300, 5, _on_zoa)

        # ── Fixed sliders ────────────────────────────────────────────────
        det_v = tk.DoubleVar(value=atype.flee_dist)
        def _on_det(val, t=atype): t.flee_dist = float(val)
        _slider_row(col, "Detect range", det_v, 10, 400, 10, _on_det)

        ss_v = tk.DoubleVar(value=atype.speed_scale)
        def _on_ss(val, t=atype): t.speed_scale = float(val)
        _slider_row(col, "Speed scale", ss_v, 0.1, 5.0, 0.1, _on_ss)

        sv_v = tk.DoubleVar(value=atype.social_speed)
        def _on_sv(val, t=atype): t.social_speed = float(val)
        _slider_row(col, "Social speed", sv_v, 0.0, 1.0, 0.05, _on_sv)

        mt_v = tk.DoubleVar(value=round(math.degrees(atype.max_turn)))
        def _on_mt(val, t=atype): t.max_turn = math.radians(float(val))
        _slider_row(col, "Max turn (°)", mt_v, 1, 180, 1, _on_mt)

        no_v = tk.DoubleVar(value=round(math.degrees(atype.noise)))
        def _on_no(val, t=atype): t.noise = math.radians(float(val))
        _slider_row(col, "Noise σ (°)", no_v, 0, 90, 1, _on_no)

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

        return col

    def _rebuild_panels():
        """Destroy all existing panels and recreate them from all_types."""
        for col in list(_panels.values()):
            col.destroy()
        _panels.clear()
        n_vars.clear()
        for t in all_types:
            _panels[t.name] = _build_panel(t)
        _count_lbl.config(text=str(len(all_types)))

    def _add_type():
        if len(all_types) >= _MAX_TYPES:
            return
        slot = next((slot for slot in _EXTRA_SLOTS
                     if slot["name"] not in {t.name for t in all_types}), None)
        if slot is None:
            slot = dict(name=f"type_{len(all_types) + 1}", color="#56fc9a")
            while slot["name"] in {t.name for t in all_types}:
                slot["name"] += "_new"
        src = all_types[-1]
        t = src.clone(slot["name"])
        t.color = slot["color"]
        t.n_default = len(model.groups.get(src.name, []))
        t.hunts.discard(t.name)
        t.flees.discard(t.name)
        all_types.append(t)
        model.agent_types[t.name] = t
        model.groups[t.name] = []
        model.init_agents(t.name, t.n_default)
        _rebuild_panels()

    def _remove_type():
        if len(all_types) <= 1:
            return
        t = all_types.pop()
        del model.agent_types[t.name]
        del model.groups[t.name]
        for remaining in all_types:
            remaining.hunts.discard(t.name)
            remaining.flees.discard(t.name)
        _rebuild_panels()

    # ================================================================== #
    # Settings collect / save
    # ================================================================== #
    def _collect_settings():
        return {
            "speed":             speed_var.get(),
            "fov":               fov_var.get(),
            "catch_dist":        catch_dist_var.get(),
            "bounded":           bounded_var.get(),
            "wall_avoid_radius": wall_ar_var.get(),
            "wall_strength":     wall_as_var.get(),
            "thigmotaxis":       wall_thig_var.get(),
            "show_edges":        show_edges_var.get(),
            "width":             config.WIDTH,
            "height":            config.HEIGHT,
            "type_count":        len(all_types),
            "mask": str(Path(mask).expanduser().resolve()) if mask is not None else None,
            "agents": {t.name: {**t.to_dict(), "n_default": n_vars[t.name].get()}
                       for t in all_types},
        }

    # ================================================================== #
    # Action buttons
    # ================================================================== #
    def reset_arena():
        stop_sim()
        model.clear()
        _step[0] = 0
        if _log is not None:
            _log.clear()
        for t in all_types:
            model.init_agents(t.name, n_vars[t.name].get())
        canvas.delete("agents")
        draw()

    def exit_sim():
        stop_sim()
        _save_settings(_collect_settings())
        try:
            root.destroy()
        except tk.TclError:
            pass

    btn_frame = tk.Frame(top)
    btn_frame.pack(fill=tk.X, pady=(0, 2))
    for lbl, cmd in [("▶ Start", start_sim), ("⏸ Pause", stop_sim),
                     ("↺ Reset", reset_arena), ("✕ Exit", exit_sim)]:
        tk.Button(btn_frame, text=lbl, command=cmd, width=8).pack(side=tk.LEFT, padx=1)

    _spf = [1]
    spf_frame = tk.Frame(top)
    spf_frame.pack(fill=tk.X, pady=(0, 4))
    tk.Label(spf_frame, text="Steps/frame:", anchor=tk.W).pack(side=tk.LEFT)
    for mult in [1, 5, 20, 100]:
        tk.Button(spf_frame, text=f"×{mult}", width=4,
                  command=lambda m=mult: _spf.__setitem__(0, m)).pack(side=tk.LEFT, padx=1)

    root.protocol("WM_DELETE_WINDOW", exit_sim)

    # ── Agent type count ────────────────────────────────────────────────
    tk.Frame(top, height=2, bg="#888888").pack(fill=tk.X, pady=3)
    count_row = tk.Frame(top)
    count_row.pack(fill=tk.X)
    tk.Label(count_row, text="Agent types:", anchor=tk.W).pack(side=tk.LEFT)
    _count_lbl = tk.Label(count_row, text=str(len(all_types)),
                          width=2, font=("", 9, "bold"))
    _count_lbl.pack(side=tk.LEFT)
    tk.Button(count_row, text="◀", width=2, command=_remove_type).pack(side=tk.LEFT)
    tk.Button(count_row, text="▶", width=2, command=_add_type).pack(side=tk.LEFT)
    tk.Frame(top, height=1, bg="#cccccc").pack(fill=tk.X, pady=2)

    # ── Global sliders ──────────────────────────────────────────────────
    tk.Frame(top, height=2, bg="#888888").pack(fill=tk.X, pady=3)
    tk.Label(top, text="Global", font=("", 9, "bold"), anchor=tk.W).pack(fill=tk.X)

    speed_var = tk.DoubleVar(value=saved.get("desired_speed", saved.get("speed", SPEED_LIMIT)))
    _slider_row(top, "Speed", speed_var, 0.5, 20, 0.5)

    fov_var = tk.DoubleVar(value=saved.get("fov_deg", saved.get("fov", FOV)))
    _slider_row(top, "Field of view (°)", fov_var, 0, 360, 10)

    catch_dist_var = tk.DoubleVar(value=saved.get("catch_dist", CATCH_DIST))
    _slider_row(top, "Catch distance", catch_dist_var, 5, 100, 5)

    # ── Arena / wall controls ───────────────────────────────────────────
    tk.Frame(top, height=2, bg="#888888").pack(fill=tk.X, pady=3)
    tk.Label(top, text="Arena", font=("", 9, "bold"), anchor=tk.W).pack(fill=tk.X)

    bounded_var = tk.BooleanVar(value=bounded)
    def _on_bounded():
        if _env is not None:
            _env.bounded = bounded_var.get()
    tk.Checkbutton(top, text="Bounded arena (canvas edges = walls)",
                   variable=bounded_var, command=_on_bounded).pack(anchor=tk.W)
    tk.Frame(top, height=1, bg="#cccccc").pack(fill=tk.X, pady=2)

    wall_ar_var = tk.DoubleVar(value=saved.get("wall_avoid_radius", 50.0))
    def _on_wall_ar(val):
        if _env is not None: _env.avoid_radius = float(val)
    _slider_row(top, "Wall avoid radius", wall_ar_var, 5, 150, 5, _on_wall_ar)

    wall_as_var = tk.DoubleVar(value=saved.get("wall_strength", 6.0))
    def _on_wall_as(val):
        if _env is not None: _env.avoid_strength = float(val)
    _slider_row(top, "Wall strength", wall_as_var, 0, 20, 0.5, _on_wall_as)

    wall_thig_var = tk.DoubleVar(value=saved.get("thigmotaxis", 0.0))
    def _on_wall_thig(val):
        if _env is not None: _env.thigmotaxis = float(val)
    _slider_row(top, "Wall follow", wall_thig_var, 0.0, 3.0, 0.05, _on_wall_thig)

    # ── Display overlays ────────────────────────────────────────────────
    tk.Frame(top, height=2, bg="#888888").pack(fill=tk.X, pady=3)
    tk.Label(top, text="Display", font=("", 9, "bold"), anchor=tk.W).pack(fill=tk.X)
    show_edges_var = tk.BooleanVar(value=saved.get("show_edges", False))
    tk.Checkbutton(top, text="Show geometric neighbours",
                   variable=show_edges_var).pack(anchor=tk.W)
    tk.Frame(top, height=1, bg="#cccccc").pack(fill=tk.X, pady=2)

    # _env assigned below after canvas size is known
    _env = None

    # ── Build initial panels ────────────────────────────────────────────
    for t in all_types:
        _panels[t.name] = _build_panel(t)

    # ================================================================== #
    # Resolve canvas size, build env + model, start
    # ================================================================== #
    root.update()
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    if w <= 1:
        w = root.winfo_screenwidth()  - right.winfo_reqwidth() - 10
        h = root.winfo_screenheight() - 80
    config.WIDTH  = w
    config.HEIGHT = h

    _env = Environment(w, h,
                       bounded=bounded_var.get(),
                       avoid_radius=wall_ar_var.get(),
                       avoid_strength=wall_as_var.get(),
                       thigmotaxis=wall_thig_var.get())
    if mask is not None:
        _env.load_mask(mask)
        _show_mask_overlay()

    model = Model(agent_types=all_types, seed=seed)
    model.env = _env
    for t in all_types:
        model.init_agents(t.name, t.n_default)

    draw()
    root.mainloop()
    return _log
