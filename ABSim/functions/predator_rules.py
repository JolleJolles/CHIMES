import random
import math
from ABSim.config import HUNTING_SIGHT, PRED_REP_WANDER_FACTOR, PRED_REP_HUNT_FACTOR
from ABSim.functions.agent_rules import move_toward_center, limit_speed, keep_within_bounds, wander, avoid_others, match_velocity
# -------------------------------
# RULE REGISTRY
# -------------------------------

def get_pred_rules(rule_names):
    result = []
    for name in rule_names:
        if name in predator_rule_groups:
            for func_name in predator_rule_groups[name]:
                func = predator_rule_names.get(func_name)
                if func:
                    result.append((func_name, func))
        elif name in predator_rule_names:
            func = predator_rule_names.get(name)
            result.append((name, func))
    return result
    

# -------------------------------
# BASIC MOVEMENT RULES
# -------------------------------
### Imported from agent_rules.py --> wander()

# -------------------------------
# PREDATOR SOCIAL RULES
# -------------------------------"
"""
def pred_repel_check(pred, preds): # will disable it soon
    repel_wander = [p for p in preds if p != pred and pred.distance(p) < pred.repel_dist]
    repel_hunt = [p for p in preds if p != pred and pred.distance(p) < pred.repel_dist_hunt]
    
    return [len(repel_wander)>0, len(repel_hunt)>0]

def pred_repel_hunt(pred, preds): # will disable it soon
    nearby_other_preds = [p for p in preds if p != pred and pred.distance(p) < pred.repel_dist_hunt]
    movex = 0
    movey = 0
    for other in nearby_other_preds:
        movex += pred.x - other.x
        movey += pred.y - other.y

    pred.dx += movex * PRED_REP_HUNT_FACTOR
    pred.dy += movey * PRED_REP_HUNT_FACTOR    
"""

def pred_avoid_others(pred, preds):
    nearby_other_preds = [p for p in preds if p != pred and pred.distance(p) < pred.repel_dist]
    movex = 0
    movey = 0
    for other in nearby_other_preds:
        movex += pred.x - other.x
        movey += pred.y - other.y

    pred.dx += movex * PRED_REP_WANDER_FACTOR
    pred.dy += movey * PRED_REP_WANDER_FACTOR


# -------------------------------
# PREDATION RULES
# -------------------------------

# not going to use check_prey anymore
def check_prey(pred, boids):
    nearby_boids = [b for b in boids if pred.distance(b) <= HUNTING_SIGHT]
    """More explicit way:
    nearby_boids = []
    for boid in boids:
        if pred.distance(boid) <= HUNTING_SIGHT:
            nearby_boids.append(boid)
    """
    if nearby_boids:
        pred.signal_state = "on"
        pred.signal_timer = 10
        return True
    else:
        pred.signal_state = "off"
        return False


def hunt(pred, boids):
    # 1. Find all boids within the hunting sight
    close_boids = [b for b in boids if pred.distance(b) < HUNTING_SIGHT]
    if not close_boids:
        return
    
    # 2. Find the centroid of the boids
    avg_x = sum(b.x for b in close_boids) / len(close_boids)        
    avg_y = sum(b.y for b in close_boids) / len(close_boids)
    
    # 3. Compute the difference b/n boid position and predator position
    move_x = avg_x - pred.x
    move_y = avg_y - pred.y
    
    # 4. Predator move toward nearest boid
    pred.dx += move_x * 0.02 
    pred.dy += move_y * 0.02
    """Note: should parameterize this factor"""

# -------------------------------
# HUNTING SIGNALING RULES -- disabled for now
# -------------------------------
"""Extra signaling related rules that could be excluded for the core model"""
"""
def check_signal(pred, preds):
    signal_preds = [p for p in preds if p.signal_state == "on" and p != pred]
    if signal_preds:
        pred.signal_pos = (signal_preds[0].x, signal_preds[0].y)
        pred.signal_state = "on"
        return True
    else:
        pred.signal_pos = []
        return False

def move_toward_signal(pred):
    if not pred.signal_pos:
        return  # no signal to respond to
    
    target_x = pred.signal_pos[0]
    target_y = pred.signal_pos[1]
    # Compute vector toward signal
    dx = target_x - pred.x
    dy = target_y - pred.y

    # Optionally scale by a factor for smooth movement toward signal
    pred.dx += dx * pred.signal_response_strength
    pred.dy += dy * pred.signal_response_strength

    # Update heading and speed if needed
    pred.head = math.atan2(pred.dy, pred.dx)
    pred.speed = math.hypot(pred.dx, pred.dy)
"""

# -------------------------------
# Refuge related rules
# -------------------------------
"""Staying away from refuge"""
def keep_away_refuge(pred, cref):
    dist_to_c = pred.distance(cref)
    if dist_to_c == 0:
        return
    
    dist_to_bound = dist_to_c - cref.r
    if dist_to_bound > cref.buff:
        return

    ux = (pred.x - cref.x) / dist_to_c
    uy = (pred.y - cref.y) / dist_to_c
    pred.dx += ux * cref.repel
    pred.dy += uy * cref.repel

# -------------------------------
# Rule registry
# -------------------------------


predator_rule_groups = {
    'pred movement rules': ['wander'],
    'pred social rules': ['move_toward_center', 'pred_avoid_others'], # Need to simplify, one repel rule is sufficient
    'predation rules': ['hunt'],
    'agent social rules': ['avoid_others', 'match_velocity', 'move_toward_center']
}

predator_rule_names = {
    'wander': wander,
    'move_toward_center': move_toward_center,
    'keep_within_bounds': keep_within_bounds,
    'pred_avoid_others': pred_avoid_others,
    'check_prey': check_prey,
    'hunt': hunt,
    'match_velocity': match_velocity,
    'avoid_others': avoid_others
}
