import random
import math
from ABSim.config import SPEED_LIMIT, FLEE_FACTOR, AVOID_FACTOR, BOID_VIS_RANGE, MATCHING_FACTOR, CATCH_DIST, TURN_FACTOR, MARGIN, WIDTH, HEIGHT, SPEED_LIMIT

def get_agent_rules(rule_names):
    result = []
    
    for name in rule_names:
        if name in agent_rule_groups:
            for func_name in agent_rule_groups[name]:
                func = agent_rule_names.get(func_name)
                if func:
                    result.append((func_name, func))
        elif name in agent_rule_names:
            func = agent_rule_names.get(name)
            result.append((name, func))
    return result
                
# -------------------------------
# Movement rules
# -------------------------------
def wander(boid):
    delta_head = random.uniform(-boid.max_turn, boid.max_turn) 
    boid.head += delta_head
    boid.dx = boid.speed * math.cos(boid.head)
    boid.dy = boid.speed * math.sin(boid.head)


def limit_speed(boid, speedlim):
    speed = math.hypot(boid.dx, boid.dy)
    if speed > speedlim:
        scale = speedlim / speed
        boid.dx *= scale
        boid.dy *= scale

def keep_within_bounds(boid): # I don't think it's currently used
    if boid.x < MARGIN:
        boid.dx += TURN_FACTOR
    if boid.x > WIDTH - MARGIN:
        boid.dx -= TURN_FACTOR
    if boid.y < MARGIN:
        boid.dy += TURN_FACTOR
    if boid.y > HEIGHT - MARGIN:
        boid.dy -= TURN_FACTOR

# -------------------------------
# Agent social rules
# -------------------------------
def avoid_others(boid, boids, avoid_dist):
    move_x = 0
    move_y = 0

    for other in boids:
        if boid != other and boid.distance(other) < avoid_dist:
            move_x += boid.x - other.x
            move_y += boid.y - other.y

    boid.dx += move_x * AVOID_FACTOR
    boid.dy += move_y * AVOID_FACTOR

def match_velocity(boid, neighbors):
    avg_dx = 0
    avg_dy = 0
    num_neighbors = 0

    for other in neighbors:
        if boid != other and boid.distance(other) < BOID_VIS_RANGE:
            avg_dx += other.dx
            avg_dy += other.dy
            num_neighbors += 1

    if num_neighbors:
        avg_dx /= num_neighbors
        avg_dy /= num_neighbors
        boid.dx += (avg_dx - boid.dx) * MATCHING_FACTOR
        boid.dy += (avg_dy - boid.dy) * MATCHING_FACTOR

def move_toward_center(agent, others): # used by both boids and predators
    center_x = 0
    center_y = 0
    num_neighbors = 0

    for other in others:
        if agent != other and agent.distance(other) < agent.cohesion_visual_range and agent.angle(other) <= agent.fov/2:
            center_x += other.x
            center_y += other.y
            num_neighbors += 1

    if num_neighbors:
        center_x /= num_neighbors
        center_y /= num_neighbors
        agent.dx += (center_x - agent.x) * agent.centering_factor
        agent.dy += (center_y - agent.y) * agent.centering_factor

# -------------------------------
# Predator avoidance related rules
# -------------------------------
def detect_preds(boid, preds, flee_dist):
    close_preds = [p for p in preds if boid.distance(p) < flee_dist]
    if close_preds:
        boid.state = "alarm"
    else:
        boid.state = "calm"     

def flee(boid, preds, flee_dist):
    if boid.state == 'alarm':
        #find the predators that are within the flee zone
        close_preds = [p for p in preds if boid.distance(p) < flee_dist]
        if not close_preds:
            return
        #calculate their average position
        avg_x = sum(p.x for p in close_preds)/ len(close_preds)
        avg_y = sum(p.y for p in close_preds)/ len(close_preds)
        
        away_x = boid.x - avg_x
        away_y = boid.y - avg_y
        
        dist = math.hypot(away_x, away_y)
        if dist > 0:
            away_x /= dist
            away_y /= dist
        
        DESIRED_FLEE_SPEED = SPEED_LIMIT * 1.05
        desired_dx = away_x * DESIRED_FLEE_SPEED
        desired_dy = away_y * DESIRED_FLEE_SPEED
        
        STEER_ALPHA = 0.15
        boid.dx += (desired_dx - boid.dx) * STEER_ALPHA * FLEE_FACTOR
        boid.dy += (desired_dy - boid.dy) * STEER_ALPHA * FLEE_FACTOR
    else:
        return

def is_eaten(boid, predators, catch_dist):
    for pred in predators:
        if boid.distance(pred) < catch_dist:
            return True
    return False

# -------------------------------
# Rule registry
# -------------------------------
agent_rule_groups = {
    'agent movement rules': ['wander'],
    'agent social rules': ['avoid_others', 'match_velocity', 'move_toward_center'],
    'pred avoid rules': ['detect_preds', 'flee']
}

agent_rule_names = {
    'wander': wander,
    'keep_within_bounds': keep_within_bounds,
    'avoid_others': avoid_others,
    'match_velocity': match_velocity,
    'move_toward_center': move_toward_center,
    'detect_preds': detect_preds,
    'flee': flee,
    'is_eaten': is_eaten
}

