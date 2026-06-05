import math
from ABSim.config import FLEE_FACTOR


def is_eaten(agent, hunters, catch_dist):
    return any(agent.distance(h) < catch_dist for h in hunters)


def detect_preds(agent, threats, dist):
    agent.state = "alarm" if any(agent.distance(t) < dist for t in threats) else "calm"


def flee(agent, threats, flee_dist, max_speed=None):
    """Steer agent away from the centroid of close threats."""
    if agent.state != "alarm":
        return
    close = [t for t in threats if agent.distance(t) < flee_dist]
    if not close:
        return
    avg_x = sum(t.x for t in close) / len(close)
    avg_y = sum(t.y for t in close) / len(close)
    away_x = agent.x - avg_x
    away_y = agent.y - avg_y
    dist = math.hypot(away_x, away_y)
    if dist > 0:
        away_x /= dist
        away_y /= dist
    if max_speed is None:
        from ABSim.config import SPEED_LIMIT
        max_speed = SPEED_LIMIT
    desired_speed = max_speed * 1.05
    agent.dx += (away_x * desired_speed - agent.dx) * 0.15 * FLEE_FACTOR
    agent.dy += (away_y * desired_speed - agent.dy) * 0.15 * FLEE_FACTOR


def hunt(agent, targets, hunt_dist):
    """Steer agent toward the centroid of targets within hunt_dist."""
    close = [t for t in targets if agent.distance(t) < hunt_dist]
    if not close:
        return
    avg_x = sum(t.x for t in close) / len(close)
    avg_y = sum(t.y for t in close) / len(close)
    agent.dx += (avg_x - agent.x) * 0.02
    agent.dy += (avg_y - agent.y) * 0.02


def limit_speed(agent, max_speed):
    speed = math.hypot(agent.dx, agent.dy)
    if speed > max_speed:
        scale = max_speed / speed
        agent.dx *= scale
        agent.dy *= scale
