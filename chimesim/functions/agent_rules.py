"""Directed between-type perception and optional capture, batched per type."""
import numpy as np
from chimesim import config


def _visible_offsets(agents, targets, distances, environment=None, fov_deg=None):
    positions = np.array([[a.x, a.y] for a in agents], dtype=float).reshape(-1, 2)
    target_positions = np.array([[a.x, a.y] for a in targets], dtype=float).reshape(-1, 2)
    delta = target_positions[None, :, :] - positions[:, None, :]
    if environment is None or not environment.bounded:
        size = np.array([environment.width, environment.height] if environment is not None
                        else [config.WIDTH, config.HEIGHT])
        delta -= np.round(delta / size) * size
    visible = np.linalg.norm(delta, axis=2) < np.asarray(distances)[..., None]
    if fov_deg is not None:
        headings = np.array([np.arctan2(a.dy, a.dx) if np.hypot(a.dx, a.dy) > 1e-10 else a.heading
                             for a in agents])
        angle = (np.arctan2(delta[:, :, 1], delta[:, :, 0]) - headings[:, None] + np.pi) % (2*np.pi) - np.pi
        visible &= (fov_deg > 0) & (np.abs(angle) <= np.deg2rad(fov_deg)/2)
    if environment is not None and visible.any():
        clear = environment.los_mask(positions, target_positions, candidates=visible)
        if clear is not None:
            visible &= clear
    return delta, visible


def relationship_direction(agents, targets, distances, environment=None, fov_deg=270, *, away=False):
    """Unit direction to/from the centroid of perceived targets and detection flags."""
    delta, visible = _visible_offsets(agents, targets, distances, environment, fov_deg)
    direction = (delta * visible[:, :, None]).sum(axis=1)
    lengths = np.linalg.norm(direction, axis=1)
    direction /= np.where(lengths > 0, lengths, 1)[:, None]
    if away:
        direction *= -1
    return direction, visible.any(axis=1)


def captured_mask(agents, captors, distance, environment=None):
    """Physical contact at the step snapshot; walls block capture, FOV does not."""
    _, visible = _visible_offsets(agents, captors, distance, environment)
    return visible.any(axis=1)
