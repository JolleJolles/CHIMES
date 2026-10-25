"""Optional individual-position history; metric calculations live in metrics.py."""
import csv
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from chimesim import metrics
from chimesim.ui_events import bind_wheel, wheel_units, choice, popup_active, show_window


_MODES = {"Spatial leadership": "front_back_relative", "Centre distance": "centroid_distance_relative"}


@lru_cache(maxsize=16)
def _palette(mode, name=None):
    stops = np.array([(49, 54, 149), (255, 255, 255), (165, 0, 38)] if mode == "Spatial leadership" else
                     [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)])
    if name:
        from chimesim.agents import _values_to_colors
        colours = _values_to_colors(np.linspace(0, 1, 256), name)
        return np.array([[int(c[i:i+2], 16) for i in (1,3,5)] for c in colours] + [[184,184,184]], dtype=np.uint8)
    positions = np.linspace(0, len(stops) - 1, 256)
    colours = np.column_stack([np.interp(positions, np.arange(len(stops)), stops[:, channel]) for channel in range(3)])
    return np.vstack((colours.astype(np.uint8), np.array([[184, 184, 184]], dtype=np.uint8)))


def _indices(values, mode, limits=None):
    values = np.asarray(values)
    low, high = limits if limits is not None else ((-1.,1.) if mode == 'Spatial leadership' else (0.,1.))
    scaled = (values-low) / (high-low)
    return np.where(np.isfinite(values), np.clip(np.nan_to_num(scaled), 0, 1) * 255, 256).astype(int)


class PositionMeans:
    """Cumulative sampled means since reset; non-finite observations do not count."""
    def __init__(self):
        self.totals = {}
        self.previous = None

    def add(self, frame):
        if self.previous is not None and self.previous['step'] == frame['step']:
            self._accumulate(self.previous, -1)
        self._accumulate(frame, 1)
        self.previous = frame
        means = {}
        for key in _MODES.values():
            means[key] = np.array([self.totals.get((int(uid),key), (0.,0))[0] /
                                  self.totals[(int(uid),key)][1]
                                  if self.totals.get((int(uid),key), (0.,0))[1] else np.nan
                                  for uid in frame['scores']['id']])
        return means

    def _accumulate(self, frame, sign):
        for key in _MODES.values():
            for uid, value in zip(frame['scores']['id'], frame['scores'][key]):
                if np.isfinite(value):
                    total, count = self.totals.get((int(uid),key), (0.,0))
                    self.totals[(int(uid),key)] = (total+sign*value, count+sign)


def calculate_position_frame(agents, environment, multiplier, step, trait_keys):
    scores = metrics.within_group_positions(agents, environment, multiplier=multiplier, largest_only=True)
    traits = np.array([[a.individual.get(key, getattr(a.type, key)) for key in trait_keys]
                       for a in agents], dtype=float).reshape(-1, len(trait_keys))
    return dict(step=step, scores=scores, traits=traits, types=[a.type.name for a in agents],
                row_colours=[a.color or a.type.color for a in agents],
                type_colours=[a.type.color for a in agents],
                collective=metrics.largest_group_metrics(agents, environment, scores['component']),
                milling=metrics.milling_by_type(agents, environment, scores['component']))


class PositionHistory:
    """Bounded, sampled history with stable agent rows and trait-based ordering."""
    def __init__(self, root, trait_labels, colour_mode="Type colours", order_by="Agent ID",
                 view="Spatial leadership", on_change=None, palette="RdBu_r", statistic="Instantaneous"):
        self.window = tk.Toplevel(root)
        self.window.title("Individual positions")
        self.trait_labels = dict(trait_labels)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chimesim-positions')
        self.pending = None
        self.poll_id = None
        self.generation = 0
        self.closed = False
        self.on_ready = None
        self.means = PositionMeans()
        self.latest_frame = None
        self.signature = None
        self.refresh_requested = False
        self.frames = deque(maxlen=180)
        self.baselines = {}
        self.last_time = -float("inf")
        self.latest = None
        self.photo = None
        self.on_change = on_change
        self.colour_limits = {'Spatial leadership': [-1.,1.], 'Centre distance': [0.,1.]}
        self.palette_var = tk.StringVar(value=palette)
        self.statistic_var = tk.StringVar(value=statistic)
        self.colour_var = tk.StringVar(value="Type colours" if colour_mode == "Traits / types" else colour_mode)
        self.mode_var = tk.StringVar(value=view if isinstance(view, str) and view in _MODES else "Spatial leadership")
        self.order_var = tk.StringVar(value=order_by if isinstance(order_by, str) and order_by in {"Agent ID", *trait_labels.values()} else "Agent ID")
        self.type_var = tk.StringVar(value="All types")
        header = tk.Frame(self.window, padx=8, pady=6)
        header.pack(fill=tk.X)
        row = tk.Frame(header)
        row.pack(fill=tk.X)
        tk.Label(row, text="Show:").pack(side=tk.LEFT)
        choice(row, self.mode_var, *_MODES, command=lambda _: self.change_view()).pack(side=tk.LEFT)
        tk.Label(row, text="Agents:").pack(side=tk.LEFT)
        choice(row, self.type_var, "All types", "A1", "A2", "A3", command=lambda _: self.render()).pack(side=tk.LEFT)
        row = tk.Frame(header)
        row.pack(fill=tk.X)
        tk.Label(row, text="Order rows by:").pack(side=tk.LEFT)
        choice(row, self.order_var, "Agent ID", *trait_labels.values(), command=lambda _: self.reorder()).pack(side=tk.LEFT)
        tk.Button(row, text="Reorder rows", command=self.reorder).pack(side=tk.LEFT)
        tk.Button(row, text="Export history", command=self.export).pack(side=tk.LEFT)
        row = tk.Frame(header)
        row.pack(fill=tk.X)
        tk.Label(row, text="Values:").pack(side=tk.LEFT)
        choice(row, self.statistic_var, "Instantaneous", "Mean since reset", command=lambda _: self.change_view()).pack(side=tk.LEFT)
        tk.Label(row, text="Means exclude missing/out-of-group samples.", fg="#777777").pack(side=tk.LEFT)
        body = tk.Frame(self.window)
        body.pack(fill=tk.BOTH, expand=True)
        self.scroll = tk.Scrollbar(body)
        self.canvas = tk.Canvas(body, bg="white", highlightthickness=0, yscrollcommand=self.scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll.config(command=self.canvas.yview)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Configure>", lambda _: self.render())
        bind_wheel(self.window, lambda event: self.canvas.yview_scroll(wheel_units(event), "units"))

    def close(self):
        self.closed = True
        if getattr(self, 'poll_id', None) is not None:
            self.window.after_cancel(self.poll_id)
            self.poll_id = None
        self.generation += 1
        self.executor.shutdown(wait=False, cancel_futures=True)

    def change_view(self):
        self.colour_var.set(self.mode_var.get())
        self.render()
        if self.on_change:
            self.on_change()

    def clear(self):
        self.generation += 1
        self.signature = None
        self.refresh_requested = True
        self.means = PositionMeans()
        self.latest_frame = None
        self.frames.clear()
        self.baselines.clear()
        self.latest = None
        self.last_time = -float("inf")
        self.render()

    def update(self, agents, environment, multiplier, step, force=False):
        if self.closed:
            return False
        changed = self._consume()
        if force:
            self.refresh_requested = True
        signature = (self.generation, step, multiplier, tuple(a.unique_id for a in agents))
        now = time.perf_counter()
        refresh = getattr(self, 'refresh_requested', False)
        if self.pending is not None or (signature == self.signature and not refresh) or (not refresh and now-self.last_time < .4):
            return changed
        self.refresh_requested = False
        self.last_time = now
        self.signature = signature
        # Copy mutable agent/type data on the UI thread; worker only reads the
        # detached snapshot and never calls Tk. At most one job is outstanding.
        types = {}
        for a in agents:
            if id(a.type) not in types:
                types[id(a.type)] = copy.copy(a.type)
        snapshot = []
        for a in agents:
            item = copy.copy(a)
            item.type = types[id(a.type)]
            item.individual = a.individual.copy()
            snapshot.append(item)
        self.pending = (self.generation, self.executor.submit(calculate_position_frame,
                        snapshot, copy.copy(environment), multiplier, step, tuple(self.trait_labels)))
        if getattr(self, 'poll_id', None) is None:
            self.poll_id = self.window.after(30, self._poll)
        return changed

    def _poll(self):
        self.poll_id = None
        if self.closed:
            return
        changed = self._consume()
        if self.on_ready and (changed or (self.pending is None and getattr(self, 'refresh_requested', False))):
            self.on_ready()
        if self.pending is not None and self.poll_id is None:
            self.poll_id = self.window.after(30, self._poll)

    def _consume(self):
        if self.pending is None or not self.pending[1].done():
            return False
        generation, future = self.pending
        self.pending = None
        frame = future.result()
        if generation != self.generation:
            return False
        self._accept(frame)
        return True

    def _accept(self, frame):
        result, traits, step = frame['scores'], frame['traits'], frame['step']
        frame['means'] = self.means.add(frame)
        if self.frames and self.frames[-1]['step'] == step:
            self.frames[-1] = frame
        else:
            self.frames.append(frame)
        self.latest = result
        self.latest_frame = frame
        for i, uid in enumerate(result['id']):
            self.baselines.setdefault(int(uid), (frame['types'][i], traits[i].copy()))
        retained = {int(uid) for observed in self.frames for uid in observed['scores']['id']}
        self.baselines = {uid: data for uid, data in self.baselines.items() if uid in retained}
        if self.window.state() != 'withdrawn' and not popup_active(self.window):
            self.render()

    def values(self, frame, mode):
        return frame['means' if self.statistic_var.get() == 'Mean since reset' else 'scores'][_MODES[mode]]

    def reorder(self):
        if self.frames:
            frame = self.frames[-1]
            for i, uid in enumerate(frame["scores"]["id"]):
                self.baselines[int(uid)] = (frame["types"][i], frame["traits"][i].copy())
        self.render()
        if self.on_change:
            self.on_change()

    def colours(self):
        if self.latest is None or self.colour_var.get() not in _MODES:
            return {}
        mode = self.colour_var.get()
        rgb = _palette(mode, self.palette_var.get())[_indices(self.values(self.latest_frame, mode), mode, self.colour_limits[mode])]
        return {int(uid): "#%02x%02x%02x" % tuple(colour) for uid, colour in zip(self.latest["id"], rgb)}

    def render(self):
        if popup_active(self.window):
            return
        self.canvas.delete("all")
        if not self.frames:
            self.canvas.create_text(12, 20, anchor="nw", text="No position samples yet.")
            return
        mode = self.mode_var.get()
        palette = _palette(mode, self.palette_var.get())
        order_key = next((key for key, label in self.trait_labels.items() if label == self.order_var.get()), None)
        column = list(self.trait_labels).index(order_key) if order_key else None
        row_ids = [uid for uid, (kind, _) in self.baselines.items()
                   if self.type_var.get() == "All types" or self.type_var.get() == kind]
        row_ids.sort(key=lambda uid: (self.baselines[uid][1][column], uid) if column is not None else (uid,))
        if not row_ids:
            self.canvas.create_text(12, 20, anchor="nw", text="No agents for this selection.")
            return
        rows = {uid: index for index, uid in enumerate(row_ids)}
        image = np.full((len(rows), len(self.frames)), 256, dtype=int)
        for frame_index, frame in enumerate(self.frames):
            indices = _indices(self.values(frame, mode), mode, self.colour_limits[mode])
            for uid, value in zip(frame["scores"]["id"], indices):
                if int(uid) in rows:
                    image[rows[int(uid)], frame_index] = value
        left, top = 90, 100
        width = max(180, self.canvas.winfo_width() - left - 22)
        height = max(len(rows), self.canvas.winfo_height() - top - 35)
        row_height = height / len(rows)
        # Resample the raster to the available space so resizing spreads rows smoothly.
        row_indices = np.minimum(np.arange(height) * len(rows) // height, len(rows) - 1)
        column_indices = np.minimum(np.arange(width) * len(self.frames) // width, len(self.frames) - 1)
        pixels = palette[image[row_indices[:, None], column_indices[None, :]]]
        data = f"P6\n{width} {height}\n255\n".encode() + pixels.tobytes()
        self.photo = tk.PhotoImage(master=self.canvas, data=data, format="PPM")
        self.canvas.create_image(left, top, anchor="nw", image=self.photo, tags="position_history")
        self.canvas.create_text(left - 10, top - 8, anchor="se", text="Individuals", fill="#777777")
        row_colours = {int(uid): colour for uid, colour in zip(self.latest_frame['scores']['id'], self.latest_frame['row_colours' if self.colour_var.get() == 'Trait colours' else 'type_colours'])}
        swatches = np.array([[int(row_colours.get(uid,'#b8b8b8')[j:j+2],16) for j in (1,3,5)] for uid in row_ids], dtype=np.uint8)
        swatch_pixels = np.repeat(swatches[row_indices, None, :], 6, axis=1)
        self.swatch_photo = tk.PhotoImage(master=self.canvas, data=f"P6\n6 {height}\n255\n".encode()+swatch_pixels.tobytes(), format='PPM')
        self.canvas.create_image(left-10, top, anchor='nw', image=self.swatch_photo, tags='trait-swatch')
        for i in np.unique(np.linspace(0, len(row_ids)-1, min(40,len(row_ids))).astype(int)):
            y = top + (i + .5) * row_height
            self.canvas.create_line(left - 3, y, left, y, fill="#888888", tags="agent_tick")
        for i in sorted({0, len(row_ids) - 1}):
            if column is None:
                label = str(i + 1)
            else:
                value = self.baselines[row_ids[i]][1][column]
                label = f"{np.degrees(value):.3g}°" if order_key in {"max_turn", "noise"} else f"{value:.3g}"
            self.canvas.create_text(left - 10, top + (i + .5) * row_height, anchor="e", text=label,
                                    font=("TkDefaultFont", 10), tags="order_endpoint")
        self.canvas.create_text(left, 10, anchor="nw", text=mode + " · " + self.statistic_var.get(), font=("TkDefaultFont", 13, "bold"))
        legend_width = min(320, width)
        for i in range(40):
            colour = "#%02x%02x%02x" % tuple(palette[round(i / 39 * 255)])
            self.canvas.create_rectangle(left + i * legend_width / 40, 38,
                                         left + (i + 1) * legend_width / 40, 49, fill=colour, outline="")
        low, high = self.colour_limits[mode]
        ends = (f"Rear {low:g}", f"Front {high:g}") if mode == "Spatial leadership" else (f"Centre {low:g}", f"Outer {high:g}")
        self.canvas.create_text(left, 57, anchor="nw", text=ends[0], fill="#777777")
        self.canvas.create_text(left + legend_width, 57, anchor="ne", text=ends[1], fill="#777777")
        bottom = top + height + 10
        self.canvas.create_text(left, bottom, anchor="nw", text=f"Step {self.frames[0]['step']}", fill="#777777")
        if len(self.frames) > 1:
            self.canvas.create_text(left + width, bottom, anchor="ne", text=str(self.frames[-1]["step"]), fill="#777777")
        self.canvas.config(scrollregion=(0, 0, left + width + 10, bottom + 22))

    def export(self):
        if not self.frames:
            return
        filename = filedialog.asksaveasfilename(parent=self.window, title="Export individual position history",
                                               initialfile="chimesim-positions.csv", defaultextension=".csv",
                                               filetypes=[("CSV position history", "*.csv")])
        if not filename:
            return
        fields = ("component", "group_size", "polarisation", "front_back", "front_back_relative",
                  "centroid_distance", "centroid_distance_relative", "centroid_x", "centroid_y")
        try:
            with open(filename, "w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(("step", "id", "type", *fields, *self.trait_labels, "mean_front_back_relative", "mean_centroid_distance_relative"))
                for frame in self.frames:
                    for i, uid in enumerate(frame["scores"]["id"]):
                        values = [frame["scores"][key][i] for key in fields]
                        writer.writerow((frame["step"], uid, frame["types"][i],
                                         *(v if np.isfinite(v) else "" for v in values), *frame["traits"][i],
                                         *(frame["means"][key][i] if np.isfinite(frame["means"][key][i]) else "" for key in _MODES.values())))
        except OSError as exc:
            messagebox.showerror("Cannot export positions", str(exc), parent=self.window)
