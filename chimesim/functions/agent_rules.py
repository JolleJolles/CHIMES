"""Cross-group interactions with consistent arena distances and wall visibility."""

import math
import numpy as np
from chimesim.config import FLEE_FACTOR
import chimesim.config as config


def _nearby(agent, targets, distance, environment=None):
    if not targets:
        return np.empty((0, 2))
    delta = np.array([[target.x - agent.x, target.y - agent.y] for target in targets], dtype=float)
    if environment is None or not environment.bounded:
        size = np.array([environment.width, environment.height] if environment is not None
                        else [config.WIDTH, config.HEIGHT])
        delta -= np.round(delta / size) * size
    keep = np.linalg.norm(delta, axis=1) < distance
    if environment is not None:
        positions = np.array([[target.x, target.y] for target in targets], dtype=float)
        visible = environment.los_mask(np.array([[agent.x, agent.y]]), positions)
        if visible is not None:
            keep &= visible[0]
    return delta[keep]


def is_eaten(agent, hunters, catch_dist, environment=None):
    return len(_nearby(agent, hunters, catch_dist, environment)) > 0


def detect_preds(agent, threats, dist, environment=None):
    agent.state = "alarm" if len(_nearby(agent, threats, dist, environment)) else "calm"


def flee(agent, threats, flee_dist, max_speed, environment=None):
    """Steer away from the centroid of nearby visible threats."""
    delta = _nearby(agent, threats, flee_dist, environment)
    if not len(delta):
        return
    away = -delta.mean(axis=0)
    length = math.hypot(*away)
    if length > 0:
        away /= length
    desired = away * max_speed * 1.05
    agent.dx += (desired[0] - agent.dx) * 0.15 * FLEE_FACTOR
    agent.dy += (desired[1] - agent.dy) * 0.15 * FLEE_FACTOR


def hunt(agent, targets, hunt_dist, environment=None):
    """Steer toward the centroid of nearby visible targets."""
    delta = _nearby(agent, targets, hunt_dist, environment)
    if len(delta):
        offset = delta.mean(axis=0)
        agent.dx += offset[0] * 0.02
        agent.dy += offset[1] * 0.02
