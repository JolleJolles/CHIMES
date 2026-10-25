"""Display the metric neighbourhood without changing social interactions."""
import math
from chimesim.model_registry import model_id


def draw_perception_circles(canvas, agents, environment, view, enabled, visual_range):
    canvas.delete('perception-circles')
    if not enabled:
        return
    scale, ox, oy = view
    tints = {}
    for agent in agents:
        t = agent.type
        if model_id(t.interaction_model) != "boids":
            continue
        radius = agent.individual.get('r_a', t.r_a)
        if not t.restricted:
            radius = min(radius, visual_range * t.visual_range_multiplier)
        if not all(math.isfinite(v) for v in (radius, agent.x, agent.y)) or radius <= 0:
            continue
        # Show the same neighbourhood on both sides of a periodic seam.
        xs = (0,) if environment.bounded else (-environment.width, 0, environment.width)
        ys = (0,) if environment.bounded else (-environment.height, 0, environment.height)
        # Tk canvas has no alpha channel. Blend against the white arena so
        # overlapping circles remain faint instead of accumulating dark dots.
        if t.color not in tints:
            rgb = canvas.winfo_rgb(t.color)
            def pale(amount):
                return '#' + ''.join(f'{round(255 * amount + channel / 257 * (1-amount)):02x}' for channel in rgb)
            tints[t.color] = pale(.98), pale(.92)
        fill, border = tints[t.color]
        r = radius * scale
        for dx in xs:
            for dy in ys:
                x, y = ox + (agent.x + dx) * scale, oy + (agent.y + dy) * scale
                if x + r < 0 or y + r < 0 or x - r > canvas.winfo_width() or y - r > canvas.winfo_height():
                    continue
                canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill, outline=border, width=.5,
                                   tags='perception-circles')
