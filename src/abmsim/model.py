# abmsim/model.py
import random
from abmsim.agents import Boid, Predator
import abmsim.config as config
from abmsim.functions.agent_rules import (
    get_agent_rules, avoid_others, match_velocity, is_eaten, detect_preds, flee, limit_speed, move_toward_center
)
from abmsim.functions.predator_rules import (get_pred_rules, hunt, check_prey, pred_avoid_others, wander, keep_away_refuge)
from abmsim.environment import cref

class Model:
    def __init__(self,
                 n_boids = None,
                 n_preds = None,
                 enable_social = None,
                 enable_classdiff = None,
                 enable_predation = None):
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
            boid_rule_names.append('agent movement rules') #Q: how do I pass a string and have it refer to a function? also need to write a wander for boids"""
        if self.enable_social and not (self.enable_classdiff or self.enable_predation):
            boid_rule_names = []
            boid_rule_names.append("agent social rules")
        if self.enable_social and self.enable_classdiff and not self.enable_predation:
            boid_rule_names = []
            boid_rule_names.append("agent social rules")
        if self.enable_social and self.enable_classdiff and self.enable_predation:
            boid_rule_names = []
            boid_rule_names.append("pred avoid rules")
            boid_rule_names.append("agent social rules")
        
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
            pred_rule_names.append('predation rules')
            pred_rule_names.append('pred social rules')
        
        self.active_pred_rules = get_pred_rules(pred_rule_names)
    
    def step(self, avoid_dist, flee_dist, catch_dist, speedlim):
        # remove eaten boids
        if self.enable_predation:
            eaten_boids = []
            for b in self.boids:
                if is_eaten(b, self.predators, catch_dist):
                    eaten_boids.append(b)
            self.boids = [b for b in self.boids if b not in eaten_boids]
        
        # boid interaction rules
        for b in self.boids:
            # Apply compiled social/movement rules
            for rule_name, rule_func in self.active_boid_rules:
                if rule_name == 'avoid_others':
                    rule_func(b, self.boids, avoid_dist)
                elif rule_name == 'detect_preds':
                    # detect predators and set boid.state -> needed before `flee`
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
        
        # predator interaction rules
        for p in self.predators:
            for rule_name, rule_func in self.active_pred_rules:
                if rule_name == 'wander':
                    rule_func(p)
                elif rule_name == 'hunt':
                    rule_func(p, self.boids)  # hunt needs boids
                elif rule_name == 'avoid_others':
                    rule_func(p,self.predators, avoid_dist)
                else:
                    rule_func(p, self.predators)

            limit_speed(p, speedlim)
            p.x = (p.x + p.dx) % config.WIDTH
            p.y = (p.y + p.dy) % config.HEIGHT
            
    def clear(self):
        self.boids.clear()
        self.predators.clear()

