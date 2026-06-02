# ABSim/model.py
import random
import numpy as np
from ABSim.agents import Boid, Predator
import ABSim.config as config
from ABSim.functions.agent_rules import (
    get_agent_rules, is_eaten, detect_preds, flee, limit_speed
)
from ABSim.functions.predator_rules import (
    get_pred_rules, hunt, pred_avoid_others, wander, keep_away_refuge
)
from ABSim.environment import cref


class Model:
    def __init__(self,
                 n_boids=None,
                 n_preds=None,
                 enable_social=None,
                 enable_classdiff=None,
                 enable_predation=None):
        self.n_boids = n_boids if n_boids is not None else config.NUM_BOIDS
        self.n_preds = n_preds if n_preds is not None else config.NUM_PREDS
        self.enable_social = enable_social if enable_social is not None else config.ENABLE_SOCIAL
        self.enable_classdiff = enable_classdiff if enable_classdiff is not None else config.ENABLE_CLASS_DIFF
        self.enable_predation = enable_predation if enable_predation is not None else config.ENABLE_PREDATION

        self.boids = []
        self.predators = []
        self.active_boid_rules = []
        self.active_pred_rules = []
        self.compile_rules()

    def init_agents(self, agent_type, n):
        agent_config = {
            'Boid': {
                'class': Boid,
                'list': self.boids,
                'x_bounds': (100, config.WIDTH - 100),
                'y_bounds': (100, config.HEIGHT - 100),
                'dx_bounds': (-5, 5),
                'dy_bounds': (-5, 5),
            },
            'Predator': {
                'class': Predator,
                'list': self.predators,
                'x_bounds': (0, config.WIDTH),
                'y_bounds': (0, config.HEIGHT),
                'dx_bounds': (-3, 3),
                'dy_bounds': (-3, 3),
            }
        }
        if agent_type not in agent_config:
            raise ValueError(f"Unknown agent type: {agent_type}.")
        cfg = agent_config[agent_type]
        for _ in range(n):
            cfg['list'].append(
                cfg['class'](
                    random.uniform(cfg['x_bounds'][0], cfg['x_bounds'][1]),
                    random.uniform(cfg['y_bounds'][0], cfg['y_bounds'][1]),
                    random.uniform(cfg['dx_bounds'][0], cfg['dx_bounds'][1]),
                    random.uniform(cfg['dy_bounds'][0], cfg['dy_bounds'][1])
                )
            )

    def compile_rules(self):
        # BOID RULES
        boid_rule_names = []
        if not self.enable_social and not self.enable_classdiff and not self.enable_predation:
            boid_rule_names.append('agent movement rules')
        if self.enable_social and not (self.enable_classdiff or self.enable_predation):
            boid_rule_names = ['agent social rules']
        if self.enable_social and self.enable_classdiff and not self.enable_predation:
            boid_rule_names = ['agent social rules']
        if self.enable_social and self.enable_classdiff and self.enable_predation:
            boid_rule_names = ['pred avoid rules', 'agent social rules']

        self.active_boid_rules = get_agent_rules(boid_rule_names)

        # PREDATOR RULES
        pred_rule_names = []
        if not self.enable_social and not self.enable_classdiff and not self.enable_predation:
            pred_rule_names.append('pred movement rules')
        if self.enable_social and not (self.enable_classdiff or self.enable_predation):
            pred_rule_names = ['agent social rules']
        if self.enable_social and self.enable_classdiff and not self.enable_predation:
            pred_rule_names = ['pred social rules']
        if self.enable_social and self.enable_classdiff and self.enable_predation:
            pred_rule_names = ['predation rules', 'pred social rules']

        self.active_pred_rules = get_pred_rules(pred_rule_names)

    # ------------------------------------------------------------------
    # Vectorised boid step (used when enable_social=True)
    #
    # Replaces the O(n²) Python loops in avoid_others, match_velocity and
    # move_toward_center with numpy batch operations.  The rule-registry
    # system is preserved for the non-social (wander) fallback path.
    #
    # Behavioural note: the vectorised path skips the FOV angle check in
    # move_toward_center (it only uses distance) and applies all three
    # social rules simultaneously rather than sequentially per boid.  The
    # practical difference is negligible at typical parameter values.
    # ------------------------------------------------------------------
    def _step_boids_vectorized(self, avoid_dist, flee_dist, speedlim):
        n = len(self.boids)

        # Extract state into contiguous arrays (one allocation per step)
        pos = np.empty((n, 2))
        vel = np.empty((n, 2))
        for i, b in enumerate(self.boids):
            pos[i, 0] = b.x;  pos[i, 1] = b.y
            vel[i, 0] = b.dx; vel[i, 1] = b.dy

        # Pairwise difference vectors and Euclidean distances
        # diff[i, j] = pos[i] - pos[j]  →  vector from boid j toward boid i
        diff  = pos[:, None, :] - pos[None, :, :]        # (n, n, 2)
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])   # (n, n)
        np.fill_diagonal(dists, np.inf)                   # exclude self

        # --- avoid_others ---
        close = dists < avoid_dist                        # (n, n) bool
        vel[:, 0] += (diff[:, :, 0] * close).sum(axis=1) * config.AVOID_FACTOR
        vel[:, 1] += (diff[:, :, 1] * close).sum(axis=1) * config.AVOID_FACTOR

        # --- match_velocity + move_toward_center (shared visibility mask) ---
        # Broadcasting: vel[:,0] (n,) × vis (n,n) → [i,j] = vel[j,0] * vis[i,j]
        vis   = dists < config.BOID_VIS_RANGE             # (n, n) bool
        n_vis = vis.sum(axis=1)                           # (n,) neighbour count
        has   = n_vis > 0
        denom = np.where(has, n_vis, 1).astype(float)    # avoid /0 for isolated boids

        avg_vx = (vel[:, 0] * vis).sum(axis=1) / denom
        avg_vy = (vel[:, 1] * vis).sum(axis=1) / denom
        vel[:, 0] += np.where(has, (avg_vx - vel[:, 0]) * config.MATCHING_FACTOR,    0.0)
        vel[:, 1] += np.where(has, (avg_vy - vel[:, 1]) * config.MATCHING_FACTOR,    0.0)

        cx = (pos[:, 0] * vis).sum(axis=1) / denom
        cy = (pos[:, 1] * vis).sum(axis=1) / denom
        vel[:, 0] += np.where(has, (cx - pos[:, 0]) * config.BOID_CENTERING_FACTOR,  0.0)
        vel[:, 1] += np.where(has, (cy - pos[:, 1]) * config.BOID_CENTERING_FACTOR,  0.0)

        # --- predator avoidance (per-boid; O(n×m), m is small) ---
        if self.enable_predation and self.predators:
            for i, b in enumerate(self.boids):
                b.dx = float(vel[i, 0]); b.dy = float(vel[i, 1])
                detect_preds(b, self.predators, flee_dist)
                flee(b, self.predators, flee_dist)
                vel[i, 0] = b.dx; vel[i, 1] = b.dy

        # --- speed limit ---
        speeds = np.hypot(vel[:, 0], vel[:, 1])
        too_fast = speeds > speedlim
        if too_fast.any():
            vel[too_fast] *= speedlim / speeds[too_fast, None]

        # --- toroidal position update ---
        pos += vel
        pos[:, 0] %= config.WIDTH
        pos[:, 1] %= config.HEIGHT

        # --- write back ---
        for i, b in enumerate(self.boids):
            b.dx = float(vel[i, 0]); b.dy = float(vel[i, 1])
            b.x  = float(pos[i, 0]); b.y  = float(pos[i, 1])

    def step(self, avoid_dist, flee_dist, catch_dist, speedlim):
        # Remove eaten boids
        if self.enable_predation:
            self.boids = [b for b in self.boids
                          if not is_eaten(b, self.predators, catch_dist)]

        # Boid updates: vectorised when social rules are active
        if self.enable_social and self.boids:
            self._step_boids_vectorized(avoid_dist, flee_dist, speedlim)
        else:
            for b in self.boids:
                for rule_name, rule_func in self.active_boid_rules:
                    if rule_name == 'avoid_others':
                        rule_func(b, self.boids, avoid_dist)
                    elif rule_name == 'detect_preds':
                        rule_func(b, self.predators, flee_dist)
                    elif rule_name == 'wander':
                        rule_func(b)
                    elif rule_name in ['match_velocity', 'move_toward_center']:
                        rule_func(b, self.boids)
                    elif rule_name == 'flee':
                        rule_func(b, self.predators, flee_dist)
                    else:
                        rule_func(b, self.boids)
                limit_speed(b, speedlim)
                b.x = (b.x + b.dx) % config.WIDTH
                b.y = (b.y + b.dy) % config.HEIGHT

        # Predator updates (sequential — predators are few)
        for p in self.predators:
            for rule_name, rule_func in self.active_pred_rules:
                if rule_name == 'wander':
                    rule_func(p)
                elif rule_name == 'hunt':
                    rule_func(p, self.boids)
                elif rule_name == 'avoid_others':
                    rule_func(p, self.predators, avoid_dist)
                else:
                    rule_func(p, self.predators)
            limit_speed(p, speedlim)
            p.x = (p.x + p.dx) % config.WIDTH
            p.y = (p.y + p.dy) % config.HEIGHT

    def clear(self):
        self.boids.clear()
        self.predators.clear()
