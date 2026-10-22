"""Body silhouettes in arena coordinates, reusable by rendering and perception.

The tail indicates heading visually; the circle defines the body's silhouette.
The perception module uses these unscaled silhouettes for agent occlusion.
"""
import math


def body_size(agent_type):
    """Per-type silhouette size; movement distances remain independent."""
    return agent_type.size * getattr(agent_type, 'size_multiplier', 1.)


def body_geometry(x, y, heading, size, style="Triangle", display_scale=1.):
    """Optional display scaling affects triangles/tails only, never circle radius."""
    ca, sa = math.cos(heading), math.sin(heading)
    if style == "Circle with tail":
        radius = size / 3
        return dict(circle=(x - radius, y - radius, x + radius, y + radius),
                    tail=(x - radius * ca, y - radius * sa,
                          x - (radius + size / 3 * display_scale) * ca,
                          y - (radius + size / 3 * display_scale) * sa))
    # Match circle diameter plus scaled tail: 2*size/3 + size/3*scale.
    size *= (2 + display_scale) / 3
    front, back, half_width = 2 * size / 3, size / 3, size / 3
    return dict(polygon=(x + front * ca, y + front * sa,
                         x - back * ca + half_width * sa, y - back * sa - half_width * ca,
                         x - back * ca - half_width * sa, y - back * sa + half_width * ca))
