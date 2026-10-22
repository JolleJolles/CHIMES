"""Automatically fitted interaction sectors with optional visual-range display."""
import math
import tkinter as tk
from chimesim.body_geometry import body_geometry


class InteractionZones:
    def __init__(self, root, snapshot, settings=None):
        self.snapshot = snapshot
        self.window = tk.Toplevel(root)
        self.window.title('Interaction zones')
        self.window.withdraw()
        self.window.minsize(240, 240)
        self.window.configure(bg='white')
        self.show_visual_range = tk.BooleanVar(value=(settings or {}).get('show_visual_range', True))
        tk.Checkbutton(self.window, text='Show visual range', variable=self.show_visual_range,
                       command=lambda: self.render(force=True), bg='white', activebackground='white',
                       selectcolor='white', highlightthickness=0).pack(anchor='w', padx=8, pady=4)
        self.canvas = tk.Canvas(self.window, bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.signature = None
        self.canvas.bind('<Configure>', lambda _: self.render(force=True))

    def settings(self):
        return {'show_visual_range': self.show_visual_range.get()}

    def render(self, force=False):
        if self.window.state() == 'withdrawn':
            return
        types, _, fov, visual_range, style, display_scale, repulsion_fov = self.snapshot()
        if not types:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        show_visual_range = self.show_visual_range.get()
        signature = (width, height, show_visual_range, fov, repulsion_fov, visual_range, style, display_scale,
                     tuple((t.name, t.r_r, t.r_o, t.r_a, t.size, t.color, t.visual_range_multiplier) for t in types))
        if not force and self.signature == signature:
            return
        self.signature = signature
        self.canvas.delete('all')
        columns = len(types) if width >= len(types) * 220 else 1
        rows = math.ceil(len(types) / columns)
        cw, ch = width / columns, height / rows
        radius = max(max(t.r_a, visual_range * t.visual_range_multiplier if show_visual_range else 0,
                         t.size * max(1, display_scale), 1) for t in types)
        scale = max(.001, min(cw, ch) - 32) / (2 * radius)
        for index, t in enumerate(types):
            cx, cy = (index % columns + .5) * cw, (index // columns + .5) * ch
            def sector(distance, colour, tag, angle=fov):
                if distance <= 0 or angle <= 0:
                    return
                r = distance * scale
                box = (cx - r, cy - r, cx + r, cy + r)
                if angle >= 360:
                    self.canvas.create_oval(*box, fill=colour, outline='#777777', tags=tag)
                else:
                    self.canvas.create_arc(*box, start=90 - angle / 2, extent=angle,
                                           style=tk.PIESLICE, fill=colour, outline='#777777', tags=tag)
            sector(t.r_a, '#e0ebf5', 'attraction')
            sector(t.r_o, '#dcebdc', 'alignment')
            sector(t.r_r, '#f1d4d4', 'repulsion', repulsion_fov)
            if show_visual_range:
                sector(visual_range * t.visual_range_multiplier, '', 'visual-range')
            for kind, values in body_geometry(0, 0, -math.pi / 2, t.size, style, display_scale=display_scale).items():
                coords = [value * scale + (cx if i % 2 == 0 else cy) for i, value in enumerate(values)]
                if kind == 'polygon':
                    self.canvas.create_polygon(*coords, fill=t.color, outline='', tags='example-agent')
                elif kind == 'circle':
                    self.canvas.create_oval(*coords, fill=t.color, outline='', tags='example-agent')
                else:
                    self.canvas.create_line(*coords, fill=t.color, width=max(1, scale) + 1, tags='example-agent')
