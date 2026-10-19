"""
Collective-behaviour metrics computed from model.groups.

All functions accept a list of Boid objects.  compute() returns a flat
dict suitable for logging (one row per simulation step).

Metrics
-------
polarisation  : mean resultant length of unit heading vectors, ∈ [0, 1].
                0 = fully random directions, 1 = perfectly aligned.
mean_nnd      : mean nearest-neighbour distance within a group (pixels),
                with toroidal boundary correction.
"""

import numpy as np
import ABSim.config as config


def polarisation(agents):
    """Mean resultant heading length (0 = disordered, 1 = aligned)."""
    n = len(agents)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    dx = np.array([a.dx for a in agents], dtype=float)
    dy = np.array([a.dy for a in agents], dtype=float)
    sp = np.hypot(dx, dy)
    ok = sp > 1e-10
    if not ok.any():
        return 0.0
    ux = (dx[ok] / sp[ok]).mean()
    uy = (dy[ok] / sp[ok]).mean()
    return float(np.hypot(ux, uy))


def mean_nnd(agents, environment=None):
    """Mean nearest-neighbour distance (pixels); pass the model environment."""
    n = len(agents)
    if n < 2:
        return 0.0
    pos  = np.array([[a.x, a.y] for a in agents], dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]          # (n, n, 2)
    if environment is None or not environment.bounded:
        width = environment.width if environment is not None else config.WIDTH
        height = environment.height if environment is not None else config.HEIGHT
        diff[:, :, 0] -= np.round(diff[:, :, 0] / width) * width
        diff[:, :, 1] -= np.round(diff[:, :, 1] / height) * height
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dists, np.inf)
    return float(dists.min(axis=1).mean())


def compute(groups, environment=None):
    """
    Compute all metrics for every group plus an overall polarisation.

    Returns a flat dict, e.g.:
        {
          'prey_n': 87, 'prey_polarisation': 0.84, 'prey_nnd': 23.1,
          'predator_n': 5, 'predator_polarisation': 0.41, 'predator_nnd': 98.3,
          'overall_polarisation': 0.79,
        }
    """
    result = {}
    all_agents = [a for agents in groups.values() for a in agents]
    for name, agents in groups.items():
        result[f"{name}_n"]            = len(agents)
        result[f"{name}_polarisation"] = polarisation(agents)
        result[f"{name}_nnd"]          = mean_nnd(agents, environment)
    result["overall_polarisation"] = polarisation(all_agents)
    return result
