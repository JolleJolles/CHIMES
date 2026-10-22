"""Snapshot visual perception, independent of movement rules and Tk.

Integrate angular silhouettes with target-centred midpoint rays. Every target
gets a minimum sample count, so small/distant bodies do not disappear between
fixed retinal bins. Occlusion is approximate and converges with resolution;
ray/body intersections and wall-grid traversal are geometric, not pixel tests.
"""
from dataclasses import dataclass
import math
import numpy as np
from chimesim.body_geometry import body_geometry, body_size


@dataclass
class Perception:
    """Directed observer-row / target-column arrays; angular values are radians."""
    ids: np.ndarray
    distance: np.ndarray
    angular_extent: np.ndarray
    in_fov_angle: np.ndarray
    visible_angle: np.ndarray
    visible_fraction: np.ndarray

    @property
    def visible(self):
        return self.visible_angle > 0


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _intersections(directions, centres, headings, sizes, styles):
    """Ray entry distances for circular or triangular bodies relative to the eye."""
    result = np.full((len(directions), len(centres)), np.inf)
    circles = np.flatnonzero(styles == 'Circle with tail')
    if len(circles):
        centre, radius = centres[circles], sizes[circles] / 3
        projection = directions @ centre.T
        discriminant = projection ** 2 - (np.sum(centre ** 2, axis=1) - radius ** 2)
        root = np.sqrt(np.maximum(0., discriminant))
        entry = np.maximum(0., projection - root)
        result[:, circles] = np.where((discriminant >= 0) & (projection + root >= 0), entry, np.inf)
    triangles = np.flatnonzero(styles == 'Triangle')
    if len(triangles):
        vertices = np.array([body_geometry(*centres[j], headings[j], sizes[j])['polygon']
                             for j in triangles]).reshape(-1, 3, 2)
        ends = np.roll(vertices, -1, axis=1)
        edges = ends - vertices
        denominator = _cross(directions[:, None, None, :], edges[None, :, :, :])
        safe = np.where(np.abs(denominator) > 1e-14, denominator, 1.)
        t = _cross(vertices, edges)[None, :, :] / safe
        u = _cross(vertices[None, :, :, :], directions[:, None, None, :]) / safe
        valid = (np.abs(denominator) > 1e-14) & (t >= 0) & (u >= -1e-12) & (u <= 1 + 1e-12)
        entry = np.min(np.where(valid, t, np.inf), axis=2)
        side = _cross(edges, -vertices)
        inside = (side >= -1e-12).all(axis=1) | (side <= 1e-12).all(axis=1)
        entry[:, inside] = 0.
        result[:, triangles] = entry
    return result


def perceive(agents, environment=None, *, body_style='Triangle', fov_deg=270.,
             max_distance=None, samples_per_body=32, angular_resolution_deg=.5,
             agent_occlusion=True):
    """Measure body visibility without changing agents or their interactions.

    body_style is a shared style or one style per agent. Physical size comes
    from body_size(agent.type); visual display scaling and decorative tails are ignored.
    fov_deg may be scalar or one value per observer. max_distance is a nonnegative scalar or per-observer
    centre-distance cutoff (None = unlimited). None environment means ordinary,
    nonperiodic space; an explicit periodic environment uses minimum-image bodies
    and shortest paths. Only one nearest image per target is considered.

    visible_fraction = visible_angle / full angular_extent, including loss to
    FOV, walls and nearer bodies. Angular extent is apparent angular width, not
    body surface area. Diagonal entries are zero (distance diagonal is zero).
    Equal-depth overlapping silhouettes do not occlude one another; overlapping
    bodies/eyes are a degenerate scene, not a collision model.
    """
    n = len(agents)
    if isinstance(samples_per_body, bool) or not isinstance(samples_per_body, int) or samples_per_body < 2:
        raise ValueError('samples_per_body must be an integer >= 2')
    if not np.isfinite(angular_resolution_deg) or angular_resolution_deg <= 0:
        raise ValueError('angular_resolution_deg must be positive and finite')
    ranges = None if max_distance is None else np.broadcast_to(np.asarray(max_distance, dtype=float), (n,))
    if ranges is not None and (not np.isfinite(ranges).all() or (ranges < 0).any()):
        raise ValueError('max_distance must be nonnegative and finite or None')
    fov = np.broadcast_to(np.asarray(fov_deg, dtype=float), (n,))
    if not np.isfinite(fov).all() or np.any((fov < 0) | (fov > 360)):
        raise ValueError('fov_deg must be between 0 and 360')
    styles = np.broadcast_to(np.asarray(body_style), (n,))
    if not np.isin(styles, ['Triangle', 'Circle with tail']).all():
        raise ValueError('Unknown body style')
    positions = np.array([[a.x, a.y] for a in agents], dtype=float).reshape(n, 2)
    sizes = np.array([body_size(a.type) for a in agents], dtype=float)
    headings = np.array([math.atan2(a.dy, a.dx) if math.hypot(a.dx, a.dy) > 1e-10 else a.heading for a in agents])
    if not np.isfinite(positions).all() or not np.isfinite(headings).all() or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('Positions/headings must be finite and body sizes positive and finite')
    ids = np.array([a.unique_id for a in agents])
    distance, extent, in_fov, visible = [np.zeros((n, n)) for _ in range(4)]
    periodic = environment is not None and not environment.bounded
    arena = np.array([environment.width, environment.height]) if periodic else None
    resolution = np.deg2rad(angular_resolution_deg)
    for observer in range(n):
        offsets = positions - positions[observer]
        if periodic:
            offsets -= np.round(offsets / arena) * arena
        dist = np.linalg.norm(offsets, axis=1)
        distance[observer] = dist
        targets = np.flatnonzero((np.arange(n) != observer) &
                                (True if max_distance is None else (ranges[observer] > 0) & (dist <= ranges[observer])))
        ray_angles, ray_weights, ray_targets = [], [], []
        for local, target in enumerate(targets):
            c, size = offsets[target], sizes[target]
            bearing = math.atan2(c[1], c[0])
            if styles[target] == 'Circle with tail':
                half = math.asin(min(1., size / 3 / max(dist[target], 1e-30)))
                low, high = (-math.pi, math.pi) if dist[target] <= size / 3 else (bearing - half, bearing + half)
            else:
                polygon = np.array(body_geometry(*c, headings[target], size)['polygon']).reshape(3, 2)
                edges = np.roll(polygon, -1, axis=0) - polygon
                side = _cross(edges, -polygon)
                if (side >= -1e-12).all() or (side <= 1e-12).all():
                    low, high = -math.pi, math.pi
                else:
                    angles = np.arctan2(polygon[:, 1], polygon[:, 0])
                    angles = bearing + (angles - bearing + math.pi) % (2 * math.pi) - math.pi
                    low, high = angles.min(), angles.max()
            width = high - low
            extent[observer, target] = width
            if width <= 0 or fov[observer] == 0:
                continue
            # Clip the silhouette to FOV before sampling; this preserves narrow
            # visible slivers instead of losing them between full-body samples.
            half_fov = np.deg2rad(fov[observer]) / 2
            for turn in (-2 * math.pi, 0., 2 * math.pi):
                start = max(low, headings[observer] - half_fov + turn)
                stop = min(high, headings[observer] + half_fov + turn)
                if stop <= start:
                    continue
                angle = stop - start
                in_fov[observer, target] += angle
                count = max(samples_per_body, int(math.ceil(angle / resolution)))
                ray_angles.extend(start + (np.arange(count) + .5) * angle / count)
                ray_weights.extend([angle / count] * count)
                ray_targets.extend([local] * count)
        if not ray_angles:
            continue
        ray_angles, ray_weights, ray_targets = map(np.asarray, (ray_angles, ray_weights, ray_targets))
        # Bound temporary ray/body matrices, even for dense groups or fine sampling.
        for first in range(0, len(ray_angles), 256):
            angles = ray_angles[first:first + 256]
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            owner = ray_targets[first:first + 256]
            hits = _intersections(directions, offsets[targets], headings[targets], sizes[targets], styles[targets])
            own_hit = hits[np.arange(len(owner)), owner]
            keep = np.isfinite(own_hit)
            if agent_occlusion:
                keep &= own_hit <= hits.min(axis=1) + 1e-10
            if environment is not None and environment.has_mask and keep.any():
                selected = np.flatnonzero(keep)
                start = np.broadcast_to(positions[observer], (len(selected), 2))
                delta = directions[selected] * own_hit[selected, None]
                keep[selected] &= environment._trace(start, delta) == 1
            np.add.at(visible[observer], targets[owner[keep]], ray_weights[first:first + 256][keep])
    fraction = np.divide(visible, extent, out=np.zeros_like(visible), where=extent > 0)
    return Perception(ids, distance, extent, in_fov, visible, np.clip(fraction, 0., 1.))
