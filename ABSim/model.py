import random
import numpy as np
import ABSim.config as config
from ABSim.agents import Boid, PREY, PREDATOR
from ABSim.functions.agent_rules import is_eaten, detect_preds, flee, hunt


class Model:
    """
    Agent-based Couzin simulation.

    Each AgentType defines its own zone radii (r_r, r_o, r_a), flee_dist,
    speed_scale, max_turn, and social roles (hunts / flees).

    Within-group behaviour: vectorised Couzin 2002 (ZOR → ZOO → ZOA).
    Cross-group schooling: agents from types that share NO adversarial
        relationship (neither hunts nor flees each other) are included in
        each other's zone computation, so they naturally school together.
    Cross-group hunt/flee: applied as a separate per-agent pass after the
        social step.
    """

    def __init__(self, agent_types=None):
        if agent_types is None:
            agent_types = [PREY, PREDATOR]
        self.agent_types: dict[str, object] = {t.name: t for t in agent_types}
        self.groups: dict[str, list[Boid]]  = {t.name: [] for t in agent_types}
        self.env = None    # set to an Environment instance to enable walls

    # ------------------------------------------------------------------ #
    # Population management
    # ------------------------------------------------------------------ #
    def init_agents(self, type_name, n):
        atype  = self.agent_types[type_name]
        agents = self.groups[type_name]
        if self.env is not None:
            x0, x1, y0, y1 = self.env.spawn_bounds
        else:
            x0, x1 = 100, config.WIDTH  - 100
            y0, y1 = 100, config.HEIGHT - 100
        new_agents = []
        for _ in range(n):
            new_agents.append(Boid(
                x=random.uniform(x0, x1),
                y=random.uniform(y0, y1),
                dx=random.uniform(-3, 3),
                dy=random.uniform(-3, 3),
                agent_type=atype,
            ))
        agents.extend(new_agents)
        atype.assign_individual(new_agents)  # sample per-agent params + colours

    @property
    def all_agents(self):
        return [a for group in self.groups.values() for a in group]

    def clear(self):
        for agents in self.groups.values():
            agents.clear()

    # ------------------------------------------------------------------ #
    # Per-group vectorised Couzin step
    #
    # agents       – boids being updated (n_self of them)
    # school_targets – boids from friendly groups included in zone
    #                computation but NOT updated here
    #
    # Zone radii, flee_dist, speed_scale, max_turn come from atype so
    # that live slider changes take effect immediately.
    # ------------------------------------------------------------------ #
    def _step_group(self, agents, atype, flee_targets, hunt_targets,
                    school_targets, desired_speed, fov_deg):
        n_self  = len(agents)
        n_total = n_self + len(school_targets)

        # Per-agent parameter arrays — fall back to type-level values when no
        # individual variation is set (individual dict is empty).
        def _arr(param, default):
            if atype.has_distributions and any(param in a.individual for a in agents):
                return np.array([a.individual.get(param, default) for a in agents])
            return np.full(n_self, default)

        eff_speeds   = desired_speed * _arr("speed_scale", atype.speed_scale)
        r_r_arr      = _arr("r_r",       atype.r_r)
        r_o_arr      = _arr("r_o",       atype.r_o)
        r_a_arr      = _arr("r_a",       atype.r_a)
        max_turn_arr = _arr("max_turn",  atype.max_turn)
        noise_arr    = _arr("noise",     atype.noise)

        # Build combined arrays: self first, school targets second
        pos = np.empty((n_total, 2))
        vel = np.empty((n_total, 2))
        for i, a in enumerate(agents + school_targets):
            pos[i, 0] = a.x;  pos[i, 1] = a.y
            vel[i, 0] = a.dx; vel[i, 1] = a.dy

        # Pairwise differences: rows = self agents, cols = all visible
        # diff[i, j] = pos[i] - pos[j]  (away-from-j direction for agent i)
        diff = pos[:n_self, None, :] - pos[None, :, :]   # (n_self, n_total, 2)
        bounded = self.env is not None and self.env.bounded
        if not bounded:
            diff[:, :, 0] -= np.round(diff[:, :, 0] / config.WIDTH)  * config.WIDTH
            diff[:, :, 1] -= np.round(diff[:, :, 1] / config.HEIGHT) * config.HEIGHT
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])   # (n_self, n_total)
        dists[np.arange(n_self), np.arange(n_self)] = np.inf  # exclude self

        # FOV mask — headings of the n_self perceiving agents
        fov_half = np.deg2rad(fov_deg) / 2
        headings = np.arctan2(vel[:n_self, 1], vel[:n_self, 0])   # (n_self,)
        angle_ij = np.arctan2(-diff[:, :, 1], -diff[:, :, 0])
        rel_ang  = (angle_ij - headings[:, None] + np.pi) % (2 * np.pi) - np.pi
        in_fov   = np.abs(rel_ang) <= fov_half                     # (n_self, n_total)

        # Line-of-sight: only applied when a mask is loaded (returns None otherwise)
        if self.env is not None:
            los = self.env.los_mask(pos[:n_self], pos)
            if los is not None:
                in_fov = in_fov & los

        # Zone masks — r_*_arr[:, None] broadcasts over (n_self, n_total) dists
        zor = (dists <  r_r_arr[:, None])                                        & in_fov
        zoo = (dists >= r_r_arr[:, None]) & (dists <  r_o_arr[:, None])         & in_fov
        zoa = (dists >= r_o_arr[:, None]) & (dists <  r_a_arr[:, None])         & in_fov

        n_zor = zor.sum(axis=1)
        n_zoo = zoo.sum(axis=1)
        n_zoa = zoa.sum(axis=1)

        safe_d      = np.where((dists == 0) | np.isinf(dists), 1.0, dists)[:, :, None]
        unit_away   =  diff / safe_d
        unit_toward = -unit_away

        # Repulsion
        rep_x = (unit_away[:, :, 0] * zor).sum(axis=1)
        rep_y = (unit_away[:, :, 1] * zor).sum(axis=1)

        # Orientation — use headings of ALL visible agents (including school targets)
        h_all = np.arctan2(vel[:, 1], vel[:, 0])
        ori_x = (np.cos(h_all)[None, :] * zoo).sum(axis=1)
        ori_y = (np.sin(h_all)[None, :] * zoo).sum(axis=1)

        # Attraction
        att_x = (unit_toward[:, :, 0] * zoa).sum(axis=1)
        att_y = (unit_toward[:, :, 1] * zoa).sum(axis=1)

        # Desired social direction; ZOR overrides ZOO+ZOA; both override wander
        has_soc = (n_zoo > 0) | (n_zoa > 0)
        soc_x = np.where(n_zoo > 0, ori_x, 0.0) + np.where(n_zoa > 0, att_x, 0.0)
        soc_y = np.where(n_zoo > 0, ori_y, 0.0) + np.where(n_zoa > 0, att_y, 0.0)

        des_x = np.where(n_zor > 0, rep_x, np.where(has_soc, soc_x, vel[:n_self, 0]))
        des_y = np.where(n_zor > 0, rep_y, np.where(has_soc, soc_y, vel[:n_self, 1]))

        # Angular noise η — applied every step to all agents (Couzin 2002).
        # Keeps paths realistically curved and prevents perfect lock-on.
        des_h = np.arctan2(des_y, des_x)
        noise = np.random.uniform(-noise_arr, noise_arr)   # per-agent η
        dh    = (des_h - headings + np.pi) % (2 * np.pi) - np.pi
        new_h = headings + np.clip(dh + noise, -max_turn_arr, max_turn_arr)

        # Apply speed (writes to vel[:n_self] only)
        # social_speed=0 → strict fixed speed; >0 → ZOO speed matching + ZOR escape boost
        alpha = atype.social_speed
        if alpha > 0:
            cur = np.hypot(vel[:n_self, 0], vel[:n_self, 1])
            cur = np.where(cur < 1e-10, eff_speeds, cur)

            # ZOO: match average speed of orientation neighbours
            speeds_all  = np.hypot(vel[:, 0], vel[:, 1])
            avg_zoo_spd = (speeds_all[None, :] * zoo).sum(axis=1) / np.maximum(n_zoo, 1)
            zoo_target  = np.where(n_zoo > 0, avg_zoo_spd, eff_speeds)

            # ZOR: escape boost proportional to crowding and social_speed
            crowd      = np.clip(n_zor / 3.0, 0.0, 1.0)
            zor_target = eff_speeds * (1.0 + alpha * crowd)

            target = np.where(n_zor > 0, zor_target, zoo_target)
            spd    = cur + alpha * 0.15 * (target - cur)
            vel[:n_self, 0] = spd * np.cos(new_h)
            vel[:n_self, 1] = spd * np.sin(new_h)
        else:
            vel[:n_self, 0] = eff_speeds * np.cos(new_h)
            vel[:n_self, 1] = eff_speeds * np.sin(new_h)

        # Cross-group flee — per-agent flee_dist and speed
        if flee_targets:
            for i, a in enumerate(agents):
                a.dx = float(vel[i, 0]); a.dy = float(vel[i, 1])
                fd = a.individual.get("flee_dist", atype.flee_dist)
                detect_preds(a, flee_targets, fd)
                flee(a, flee_targets, fd, float(eff_speeds[i]))
                vel[i, 0] = a.dx; vel[i, 1] = a.dy

        # Cross-group hunt — per-agent attraction radius
        if hunt_targets:
            for i, a in enumerate(agents):
                a.dx = float(vel[i, 0]); a.dy = float(vel[i, 1])
                hunt(a, hunt_targets, float(r_a_arr[i]))
                vel[i, 0] = a.dx; vel[i, 1] = a.dy

        # Wall avoidance forces (applied alongside flee/hunt, before renorm)
        if self.env is not None:
            self.env.apply_forces(pos[:n_self], vel[:n_self],
                                  config.WIDTH, config.HEIGHT)

        # Speed renormalisation after cross-group forces
        sp    = np.hypot(vel[:n_self, 0], vel[:n_self, 1])
        sp    = np.maximum(sp, 1e-10)
        if alpha > 0:
            cap   = eff_speeds * (1.0 + alpha * 1.5)
            scale = np.where(sp > cap, cap / sp, 1.0)
        else:
            scale = eff_speeds / sp                    # strict per-agent fixed speed
        vel[:n_self, 0] *= scale
        vel[:n_self, 1] *= scale

        # Position update — bounded (clamp to canvas) or toroidal (wrap)
        new_x = pos[:n_self, 0] + vel[:n_self, 0]
        new_y = pos[:n_self, 1] + vel[:n_self, 1]
        if self.env is not None and self.env.bounded:
            pos[:n_self, 0] = np.clip(new_x, 0, config.WIDTH  - 1)
            pos[:n_self, 1] = np.clip(new_y, 0, config.HEIGHT - 1)
        else:
            pos[:n_self, 0] = new_x % config.WIDTH
            pos[:n_self, 1] = new_y % config.HEIGHT
        for i, a in enumerate(agents):
            a.dx = float(vel[i, 0]); a.dy = float(vel[i, 1])
            a.x  = float(pos[i, 0]); a.y  = float(pos[i, 1])

    # ------------------------------------------------------------------ #
    # Main step
    # ------------------------------------------------------------------ #
    def step(self, catch_dist, desired_speed, fov_deg=270):
        # Remove caught agents
        for type_name in list(self.groups):
            hunters = [
                a
                for tn, t in self.agent_types.items()
                if type_name in t.hunts
                for a in self.groups.get(tn, [])
            ]
            if hunters:
                self.groups[type_name] = [
                    a for a in self.groups[type_name]
                    if not is_eaten(a, hunters, catch_dist)
                ]

        # Per-group Couzin step
        for type_name, agents in self.groups.items():
            if not agents:
                continue
            atype = self.agent_types[type_name]

            flee_targets = [a for fn in atype.flees for a in self.groups.get(fn, [])]
            hunt_targets = [a for hn in atype.hunts for a in self.groups.get(hn, [])]

            # School targets: agents from groups with no adversarial relationship
            school_targets = [
                a
                for other_name, other_agents in self.groups.items()
                if (other_name != type_name
                    and other_name not in atype.hunts
                    and other_name not in atype.flees
                    and type_name not in self.agent_types[other_name].hunts
                    and type_name not in self.agent_types[other_name].flees)
                for a in other_agents
            ]

            self._step_group(
                agents, atype, flee_targets, hunt_targets, school_targets,
                desired_speed, fov_deg,
            )
