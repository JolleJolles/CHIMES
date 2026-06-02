import math
import numpy as np
"""
Figure out how to import parameters from config and what parameters are needed
"""
from ABSim.config import BOID_VIS_RANGE, BOID_CENTERING_FACTOR, HUNTING_SIGHT, PRED_COHESION_VIS_RANGE, PRED_CENTERING_FACTOR, DFLT_PRED_AVOID_DIST


# -------------------------------
# CLASS DEFINITIONS
# -------------------------------
class Boid:
    def __init__(self, x, y, dx, dy, fov = 0.75*np.pi):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.fov = fov
        self.head = math.atan2(self.dy, self.dx)
        self.speed = math.hypot(self.dx, self.dy)
        self.max_turn = math.pi * 0.2
        self.max_turn_flee = math.pi * 0.3
        self.cohesion_visual_range = BOID_VIS_RANGE
        self.centering_factor = BOID_CENTERING_FACTOR
        self.history = []
        self.state = "calm"
        """Maybe could add a state variable (roaming/alert)"""

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def angle(self, other):
        theta = math.atan2(other.y-self.y, other.x-self.x)
        # might need a way to normalize the angles
        diff = self.head - theta
        norm_abs_diff = abs((diff+math.pi)%(2*math.pi)-math.pi)
        return norm_abs_diff 
    def draw(self, canvas): # Presumably need to pass a canvas argument into this method
        angle = math.atan2(self.dy, self.dx)
        x, y = self.x, self.y
        p1 = (x, y)
        p2 = (x - 15 * math.cos(angle) + 5 * math.sin(angle),
              y - 15 * math.sin(angle) - 5 * math.cos(angle))
        p3 = (x - 15 * math.cos(angle) - 5 * math.sin(angle),
              y - 15 * math.sin(angle) + 5 * math.cos(angle))
        canvas.create_polygon([p1, p2, p3], fill="#558cf4", outline="", tags = "agents")
        
# vectorized version
"""    def angle_to_others(self, others):
        heading = math.atan2(self.dy, self.dx)
        others_pos = np.array([[o.x, o.y] for o in others]) # shape = (N,2)
        
        #This is a list comprehension, make an array from a list of lists
        #more explicit version:
        #`positions = []
        #for o in others:
        #    positions.append([o.x, o.y])
        
        others_pos = np.array(positions)`
        my_pos = np.array([self.x, self.y])
        diff = my_pos - others_pos
        theta = np.arctan2(diff[:,1], diff[:,0])
        
        angle_diff = heading - theta
        norm_abs_angles = (angle_diff + np.pi)
        
        in_fov = np.abs(norm_abs_angles) <= (self.fov/2)
        return in_fov"""
    


class Predator(Boid):
    def __init__(self, x, y, dx, dy, fov = 0.5*np.pi):
        super().__init__(x, y, dx, dy, fov)
        self.move_state = "wander"
        self.signal_state = "off"
        self.signal_timer = 0
        self.signal_pos = []
        self.signal_response_strength = 0.02
        self.max_turn = math.pi * 0.15
        self.repel_dist = DFLT_PRED_AVOID_DIST
        self.repel_dist_hunt = HUNTING_SIGHT*2
        self.cohesion_visual_range = PRED_COHESION_VIS_RANGE
        self.centering_factor = PRED_CENTERING_FACTOR
        # wander speed multiplier
        # hunting speed multiplier
    def draw(self, canvas): # Presumably need to pass a canvas argument into this method
        angle = math.atan2(self.dy, self.dx)
        x, y = self.x, self.y
        p1 = (x, y)
        p2 = (x - 20 * math.cos(angle) + 7 * math.sin(angle),
              y - 20 * math.sin(angle) - 7 * math.cos(angle))
        p3 = (x - 20 * math.cos(angle) - 7 * math.sin(angle),
              y - 20 * math.sin(angle) + 7 * math.cos(angle))
        canvas.create_polygon([p1, p2, p3], fill="#d94b4b", outline="", tags = "agents")
        
        # draw signal halo if signal on
        #if self.signal_timer > 0:
        #    radius = HUNTING_SIGHT
        #    alpha = int(255 * (self.signal_timer /10)) # fade out over 10 frames???
        #    canvas.create_oval(
        #        x - radius, y - radius,
        #        x + radius, y + radius,
        #        outline = "yellow", width = 2,
        #        tags = "agents"
        #    )        
        #    self.signal_timer -= 1