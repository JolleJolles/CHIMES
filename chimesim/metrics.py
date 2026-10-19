"""Collective and individual position metrics, independent of the GUI.

compute() summarises named agent types; spatial_groups() identifies proximity
components across the supplied agents. within_group_positions() measures each
agent relative to its physical component, not its type's global centroid.
Distances use model pixels and speeds pixels per step. Periodic geometry and
nearest-neighbour searches are shared across all spatial measures.
"""

import numpy as np
import chimesim.config as config
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _periodic_size(environment):
    if environment is not None and environment.bounded:
        return None
    return np.array([environment.width, environment.height] if environment is not None
                    else [config.WIDTH, config.HEIGHT], dtype=float)


def _position_tree(agents, environment):
    positions = np.array([[a.x, a.y] for a in agents], dtype=float).reshape(-1, 2)
    size = _periodic_size(environment)
    return cKDTree(positions if size is None else positions % size, boxsize=size)


def _mean_heading(velocities):
    speed = np.linalg.norm(velocities, axis=1)
    moving = speed > 1e-10
    heading = (velocities[moving] / speed[moving, None]).mean(axis=0) if moving.any() else np.zeros(2)
    return heading, float(np.linalg.norm(heading))


def polarisation(agents):
    """Mean resultant heading length (0 = disordered, 1 = aligned)."""
    n = len(agents)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    return _mean_heading(np.array([[a.dx, a.dy] for a in agents], dtype=float))[1]


def mean_nnd(agents, environment=None):
    """Mean nearest-neighbour distance (pixels); pass the model environment."""
    n = len(agents)
    if n < 2:
        return 0.0
    tree = _position_tree(agents, environment)
    return float(tree.query(tree.data, k=2)[0][:, 1].mean())


def mean_speed(agents):
    """Mean velocity magnitude in model pixels per step; zero for an empty type."""
    return float(np.mean([np.hypot(a.dx, a.dy) for a in agents])) if agents else 0.


def compute(groups, environment=None):
    """
    Summarise each named type (count, polarisation, NND, mean speed) and overall polarisation.

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
        result[f"{name}_speed"]        = mean_speed(agents)
    result["overall_polarisation"] = polarisation(all_agents)
    return result


def spatial_groups(agents, environment=None, link_distance=None, multiplier=2.5, return_labels=False):
    """Fast exploratory connectivity: groups have >=2 agents; isolates are separate.

    By default link agents within multiplier times the median positive nearest-neighbour
    distance. This heuristic has no biological calibration. Use minimum-image
    distances in periodic arenas; solid walls block direct connections.
    With return_labels=True, add a labels array aligned with agents; singletons
    have label -1. Component labels are local to each snapshot.
    """
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Grouping multiplier must be finite and positive")
    if link_distance is not None and (not np.isfinite(link_distance) or link_distance < 0):
        raise ValueError("Link distance must be finite and nonnegative")
    n = len(agents)
    if n < 2:
        result = dict(groups=0, isolated=n, link_distance=0.)
        if return_labels:
            result["labels"] = np.full(n, -1, dtype=int)
        return result
    tree = _position_tree(agents, environment)
    if link_distance is None:
        nearest = tree.query(tree.data, k=2)[0][:, 1]
        positive = nearest[nearest > 0]
        link_distance = multiplier * float(np.median(positive)) if len(positive) else 0.
    pairs = tree.query_pairs(link_distance, output_type="ndarray")
    if len(pairs) and environment is not None and environment.has_mask:
        start = tree.data[pairs[:, 0]]
        delta = tree.data[pairs[:, 1]] - start
        if not environment.bounded:
            size = np.array([environment.width, environment.height])
            delta -= np.round(delta / size) * size
        pairs = pairs[environment._trace(start, delta) == 1]
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    result = dict(groups=int(np.sum(sizes >= 2)), isolated=int(np.sum(sizes == 1)),
                  link_distance=float(link_distance))
    if return_labels:
        result["labels"] = np.where(sizes[labels] >= 2, labels, -1)
    return result


def within_group_positions(agents, environment=None, multiplier=2.5, link_distance=None,
                           min_polarisation=.2):
    """Per-agent spatial leadership and radial distance within physical clusters.

    Component labels are local to this snapshot, not persistent group identities.
    Front/back is displacement from centroid projected onto mean unit heading;
    positive is front. Relative front/back divides by the group's maximum absolute
    projection; relative centre distance divides by its maximum radial distance.
    Scores describe spatial position, not causal influence on neighbours.

    Singletons have NaN position scores. Front/back is also NaN for stationary or
    weakly aligned groups. Periodic groups are unwrapped around a circular centre
    and recentered arithmetically. An extent >= half the arena along either axis,
    or an ambiguous circular centre, yields NaN rather than a misleading centroid.
    """
    if not np.isfinite(min_polarisation) or not 0 <= min_polarisation <= 1:
        raise ValueError("min_polarisation must lie in [0, 1]")
    clustering = spatial_groups(agents, environment, link_distance, multiplier, return_labels=True)
    n = len(agents)
    labels = clustering["labels"]
    result = dict(id=np.array([a.unique_id for a in agents], dtype=int), component=labels,
                  group_size=np.ones(n, dtype=int), polarisation=np.full(n, np.nan),
                  front_back=np.full(n, np.nan), front_back_relative=np.full(n, np.nan),
                  centroid_distance=np.full(n, np.nan), centroid_distance_relative=np.full(n, np.nan),
                  centroid_x=np.full(n, np.nan), centroid_y=np.full(n, np.nan),
                  groups=clustering["groups"], isolated=clustering["isolated"],
                  link_distance=clustering["link_distance"])
    positions = np.array([[a.x, a.y] for a in agents], dtype=float).reshape(-1, 2)
    velocities = np.array([[a.dx, a.dy] for a in agents], dtype=float).reshape(-1, 2)
    size = _periodic_size(environment)
    for component in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == component)
        result["group_size"][indices] = len(indices)
        local = positions[indices]
        if size is None:
            centre = local.mean(axis=0)
            offsets = local - centre
        else:
            angles = local / size * (2 * np.pi)
            vector = np.mean(np.exp(1j * angles), axis=0)
            if np.any(np.abs(vector) < .05):
                continue
            centre = (np.angle(vector) % (2 * np.pi)) / (2 * np.pi) * size
            offsets = (local - centre + size / 2) % size - size / 2
            if np.any(np.ptp(offsets, axis=0) >= size / 2):
                continue
            correction = offsets.mean(axis=0)
            offsets -= correction
            centre = (centre + correction) % size
        result["centroid_x"][indices], result["centroid_y"][indices] = centre
        distance = np.linalg.norm(offsets, axis=1)
        result["centroid_distance"][indices] = distance
        result["centroid_distance_relative"][indices] = distance / max(distance.max(), 1e-10)
        heading, alignment = _mean_heading(velocities[indices])
        result["polarisation"][indices] = alignment
        if alignment < max(min_polarisation, 1e-10):
            continue
        front = offsets @ (heading / alignment)
        result["front_back"][indices] = front
        result["front_back_relative"][indices] = front / max(np.max(np.abs(front)), 1e-10)
    return result
