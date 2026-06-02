import tkinter as tk
import math
import random
import time
from ABSim.model import Model
import ABSim.config as config
from ABSim.config import DFLT_AVOID_DIST, DFLT_FLEE_DIST, SPEED_LIMIT, CATCH_DIST


def run(enable_social=None, enable_classdiff=None, enable_predation=None):
    root = tk.Tk()
    root.title("Boids Simulation (working title)")

    # Control panel — fixed strip on the right
    ctrlpnl = tk.Frame(root)
    ctrlpnl.pack(side=tk.RIGHT, fill=tk.Y)

    # Canvas — fills all remaining space; no fixed width/height so it adapts to
    # whatever area is left after the control panel is laid out.
    canvas = tk.Canvas(root, bg="white")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Keep config.WIDTH/HEIGHT in sync whenever the canvas is resized
    # (fires on startup, window resize, and after reset_arena redraws).
    # model.step() and init_agents() both read config at call time, so
    # wrapping bounds and spawn bounds update automatically.
    def _on_canvas_resize(event):
        config.WIDTH  = event.width
        config.HEIGHT = event.height

    canvas.bind("<Configure>", _on_canvas_resize)

    # Maximise on startup so the canvas uses the full available screen.
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

    running = {"id": None}
    _fps = [time.perf_counter(), 0.0]  # [last_frame_time, smoothed_fps]

    # ------------------------------------------------------------------ #
    # Simulation functions — defined before `model` exists; Python closures
    # resolve names at call time, so they find `model` once it is assigned
    # below (before mainloop starts).
    # ------------------------------------------------------------------ #

    def draw():
        canvas.delete("agents")
        for b in model.boids:
            angle = math.atan2(b.dy, b.dx)
            x, y = b.x, b.y
            p1 = (x, y)
            p2 = (x - 15 * math.cos(angle) + 5 * math.sin(angle),
                  y - 15 * math.sin(angle) - 5 * math.cos(angle))
            p3 = (x - 15 * math.cos(angle) - 5 * math.sin(angle),
                  y - 15 * math.sin(angle) + 5 * math.cos(angle))
            canvas.create_polygon([p1, p2, p3], fill="#56b1fc", outline="", tags="agents")
        for p in model.predators:
            angle = math.atan2(p.dy, p.dx)
            x, y = p.x, p.y
            p1 = (x, y)
            p2 = (x - 20 * math.cos(angle) + 7 * math.sin(angle),
                  y - 20 * math.sin(angle) - 7 * math.cos(angle))
            p3 = (x - 20 * math.cos(angle) - 7 * math.sin(angle),
                  y - 20 * math.sin(angle) + 7 * math.cos(angle))
            canvas.create_polygon([p1, p2, p3], fill="#fc6056", outline="", tags="agents")

    def loop():
        now = time.perf_counter()
        dt = now - _fps[0]
        _fps[0] = now
        if dt > 0:
            _fps[1] = 0.8 * _fps[1] + 0.2 * (1.0 / dt)

        model.step(avoid_var.get(), flee_var.get(), catch_dist_var.get(), speedlim_var.get())
        draw()

        canvas.delete("fps_counter")
        canvas.create_text(8, 8, anchor=tk.NW, text=f"{_fps[1]:.0f} fps",
                           fill="#999999", font=("", 10), tags="fps_counter")

        running["id"] = canvas.after(16, loop)

    def start_sim():
        if running["id"] is None:
            loop()

    def stop_sim():
        if running["id"] is not None:
            canvas.after_cancel(running["id"])
            running["id"] = None

    def _set_agent_count(agent_list, agent_type, new_n):
        current_n = len(agent_list)
        if new_n > current_n:
            model.init_agents(agent_type, new_n - current_n)
        elif new_n < current_n:
            keep = set(random.sample(range(current_n), new_n))
            if agent_type == 'Boid':
                model.boids = [b for i, b in enumerate(agent_list) if i in keep]
            else:
                model.predators = [p for i, p in enumerate(agent_list) if i in keep]

    def on_boid_count_change(val):
        _set_agent_count(model.boids, 'Boid', int(float(val)))

    def on_pred_count_change(val):
        _set_agent_count(model.predators, 'Predator', int(float(val)))

    def reset_arena():
        stop_sim()
        model.clear()
        model.init_agents('Boid', num_boids_var.get())
        canvas.delete("agents")
        draw()

    def release_predators():
        model.predators.clear()
        model.init_agents('Predator', num_preds_var.get())

    # ------------------------------------------------------------------ #
    # Control panel widgets
    # ------------------------------------------------------------------ #

    def _nudge(var, delta, lo, hi, on_change=None):
        new_val = max(lo, min(hi, var.get() + delta))
        var.set(new_val)
        if on_change is not None:
            on_change(new_val)

    def _slider_row(label, var, lo, hi, res, on_change=None):
        """Label + [◀] slider [▶] row, with optional live callback."""
        tk.Label(ctrlpnl, text=label).pack(anchor=tk.W)
        row = tk.Frame(ctrlpnl)
        row.pack(anchor=tk.W)
        tk.Button(row, text="◀", width=2,
                  command=lambda: _nudge(var, -res, lo, hi, on_change)).pack(side=tk.LEFT)
        kwargs = dict(from_=lo, to=hi, resolution=res,
                      orient=tk.HORIZONTAL, variable=var)
        if on_change:
            kwargs["command"] = on_change
        tk.Scale(row, **kwargs).pack(side=tk.LEFT)
        tk.Button(row, text="▶", width=2,
                  command=lambda: _nudge(var, +res, lo, hi, on_change)).pack(side=tk.LEFT)
        tk.Frame(ctrlpnl, height=1, bg="gray").pack(fill=tk.X, padx=10, pady=5)

    num_boids_var = tk.IntVar(value=config.NUM_BOIDS)
    _slider_row("N Boids",           num_boids_var, 0,  500, 1, on_boid_count_change)

    num_preds_var = tk.IntVar(value=config.NUM_PREDS)
    _slider_row("N Predators",       num_preds_var, 0,  30,  1, on_pred_count_change)

    avoid_var      = tk.DoubleVar(value=DFLT_AVOID_DIST)
    _slider_row("Boid avoid dist",   avoid_var,     5,  100, 1)

    flee_var       = tk.DoubleVar(value=DFLT_FLEE_DIST)
    _slider_row("Boid flee dist",    flee_var,      50, 300, 5)

    speedlim_var   = tk.DoubleVar(value=SPEED_LIMIT)
    _slider_row("Speed limit",       speedlim_var,  1,  100, 1)

    catch_dist_var = tk.DoubleVar(value=CATCH_DIST)
    _slider_row("Catch distance",    catch_dist_var, 10, 100, 5)

    def exit_sim():
        stop_sim()
        root.destroy()

    tk.Button(ctrlpnl, text="Start simulation",  command=start_sim).pack(anchor=tk.W)
    tk.Button(ctrlpnl, text="Pause simulation",  command=stop_sim).pack(anchor=tk.W)
    tk.Button(ctrlpnl, text="Reset arena",       command=reset_arena).pack(anchor=tk.W)
    tk.Button(ctrlpnl, text="Release predators", command=release_predators).pack(anchor=tk.W)
    tk.Button(ctrlpnl, text="Exit",              command=exit_sim).pack(anchor=tk.W)

    # ------------------------------------------------------------------ #
    # Resolve actual canvas dimensions after the layout has been computed.
    # root.update() processes all pending geometry events so winfo_width/
    # height return the real rendered size rather than the unmapped default
    # of 1.  Fall back to a screen-based estimate on the rare case where the
    # window has not been mapped yet.
    # ------------------------------------------------------------------ #
    root.update()
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    if w <= 1:
        w = root.winfo_screenwidth() - 260
        h = root.winfo_screenheight() - 80
    config.WIDTH  = w
    config.HEIGHT = h

    # Create model now that the correct arena bounds are known
    model = Model(
        enable_social=enable_social,
        enable_classdiff=enable_classdiff,
        enable_predation=enable_predation,
    )
    model.init_agents('Boid', config.NUM_BOIDS)

    draw()
    root.mainloop()
