"""Optional individual-position history; metric calculations live in metrics.py."""
import csv
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from chimesim import metrics
from chimesim.ui_events import bind_wheel, wheel_units, choice, popup_active, show_window


_MODES = {"Spatial leadership": "front_back_relative", "Centre distance": "centroid_distance_relative"}


def _palette(mode):
    stops = np.array([(49, 54, 149), (255, 255, 255), (165, 0, 38)] if mode == "Spatial leadership" else
                     [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)])
    positions = np.linspace(0, len(stops) - 1, 256)
    colours = np.column_stack([np.interp(positions, np.arange(len(stops)), stops[:, channel]) for channel in range(3)])
    return np.vstack((colours.astype(np.uint8), np.array([[184, 184, 184]], dtype=np.uint8)))


def _indices(values, mode):
    values = np.asarray(values)
    scaled = (values + 1) / 2 if mode == "Spatial leadership" else values
    return np.where(np.isfinite(values), np.clip(np.nan_to_num(scaled), 0, 1) * 255, 256).astype(int)


class PositionHistory:
    """Bounded, sampled history with stable agent rows and trait-based ordering."""
    def __init__(self, root, trait_labels, colour_mode="Traits / types", order_by="Agent ID",
                 view="Spatial leadership", on_change=None):
        self.window = tk.Toplevel(root)
        self.window.title("Individual positions")
        self.trait_labels = dict(trait_labels)
        self.frames = deque(maxlen=180)
        self.baselines = {}
        self.last_time = -float("inf")
        self.latest = None
        self.photo = None
        self.on_change = on_change
        self.colour_var = tk.StringVar(value=colour_mode)
        self.mode_var = tk.StringVar(value=view if isinstance(view, str) and view in _MODES else "Spatial leadership")
        self.order_var = tk.StringVar(value=order_by if isinstance(order_by, str) and order_by in {"Agent ID", *trait_labels.values()} else "Agent ID")
        self.type_var = tk.StringVar(value="All types")
        header = tk.Frame(self.window, padx=8, pady=6)
        header.pack(fill=tk.X)
        row = tk.Frame(header)
        row.pack(fill=tk.X)
        tk.Label(row, text="Show:").pack(side=tk.LEFT)
        choice(row, self.mode_var, *_MODES, command=lambda _: self.render()).pack(side=tk.LEFT)
        tk.Label(row, text="Agents:").pack(side=tk.LEFT)
        choice(row, self.type_var, "All types", "A1", "A2", "A3", command=lambda _: self.render()).pack(side=tk.LEFT)
        row = tk.Frame(header)
        row.pack(fill=tk.X)
        tk.Label(row, text="Order rows by:").pack(side=tk.LEFT)
        choice(row, self.order_var, "Agent ID", *trait_labels.values(), command=lambda _: self.reorder()).pack(side=tk.LEFT)
        tk.Button(row, text="Reorder rows", command=self.reorder).pack(side=tk.LEFT)
        tk.Button(row, text="Export history", command=self.export).pack(side=tk.LEFT)
        body = tk.Frame(self.window)
        body.pack(fill=tk.BOTH, expand=True)
        self.scroll = tk.Scrollbar(body)
        self.canvas = tk.Canvas(body, bg="white", highlightthickness=0, yscrollcommand=self.scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll.config(command=self.canvas.yview)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Configure>", lambda _: self.render())
        bind_wheel(self.window, lambda event: self.canvas.yview_scroll(wheel_units(event), "units"))

    def clear(self):
        self.frames.clear()
        self.baselines.clear()
        self.latest = None
        self.last_time = -float("inf")
        self.render()

    def update(self, agents, environment, multiplier, step, force=False):
        now = time.perf_counter()
        if not force and now - self.last_time < .4:
            return False
        self.last_time = now
        result = metrics.within_group_positions(agents, environment, multiplier=multiplier, largest_only=True)
        traits = np.array([[a.individual.get(key, getattr(a.type, key)) for key in self.trait_labels]
                           for a in agents], dtype=float).reshape(-1, len(self.trait_labels))
        frame = dict(step=step, scores=result, traits=traits, types=[a.type.name for a in agents])
        if self.frames and self.frames[-1]["step"] == step:
            self.frames[-1] = frame
        else:
            self.frames.append(frame)
        self.latest = result
        for i, uid in enumerate(result["id"]):
            self.baselines.setdefault(int(uid), (frame["types"][i], traits[i].copy()))
        retained = {int(uid) for observed in self.frames for uid in observed["scores"]["id"]}
        self.baselines = {uid: data for uid, data in self.baselines.items() if uid in retained}
        if self.window.state() != "withdrawn" and not popup_active(self.window):
            self.render()
        return True

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
        rgb = _palette(mode)[_indices(self.latest[_MODES[mode]], mode)]
        return {int(uid): "#%02x%02x%02x" % tuple(colour) for uid, colour in zip(self.latest["id"], rgb)}

    def render(self):
        if popup_active(self.window):
            return
        self.canvas.delete("all")
        if not self.frames:
            self.canvas.create_text(12, 20, anchor="nw", text="No position samples yet.")
            return
        mode = self.mode_var.get()
        palette = _palette(mode)
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
            indices = _indices(frame["scores"][_MODES[mode]], mode)
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
        for i in range(len(row_ids)):
            y = top + (i + .5) * row_height
            self.canvas.create_line(left - 5, y, left, y, fill="#888888", tags="agent_tick")
        for i in sorted({0, len(row_ids) - 1}):
            if column is None:
                label = str(i + 1)
            else:
                value = self.baselines[row_ids[i]][1][column]
                label = f"{np.degrees(value):.3g}°" if order_key in {"max_turn", "noise"} else f"{value:.3g}"
            self.canvas.create_text(left - 10, top + (i + .5) * row_height, anchor="e", text=label,
                                    font=("TkDefaultFont", 10), tags="order_endpoint")
        self.canvas.create_text(left, 10, anchor="nw", text=mode, font=("TkDefaultFont", 13, "bold"))
        legend_width = min(320, width)
        for i in range(40):
            colour = "#%02x%02x%02x" % tuple(palette[round(i / 39 * 255)])
            self.canvas.create_rectangle(left + i * legend_width / 40, 38,
                                         left + (i + 1) * legend_width / 40, 49, fill=colour, outline="")
        ends = ("Rear −1", "Front +1") if mode == "Spatial leadership" else ("Centroid 0", "Outermost 1")
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
                writer.writerow(("step", "id", "type", *fields, *self.trait_labels))
                for frame in self.frames:
                    for i, uid in enumerate(frame["scores"]["id"]):
                        values = [frame["scores"][key][i] for key in fields]
                        writer.writerow((frame["step"], uid, frame["types"][i],
                                         *(v if np.isfinite(v) else "" for v in values), *frame["traits"][i]))
        except OSError as exc:
            messagebox.showerror("Cannot export positions", str(exc), parent=self.window)
