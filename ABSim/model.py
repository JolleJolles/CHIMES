import random
import numpy as np
from scipy.spatial import Delaunay, QhullError
import ABSim.config as config
from ABSim.agents import Boid, PREY, PREDATOR
from ABSim.functions.agent_rules import is_eaten, detect_preds, flee, hunt


class Model:
    """
    Agent-based collective-behaviour simulation.

    Interaction models (set via AgentType.interaction_model):
      "zonal"   — Couzin 2002 zone-based rules (ZOR / ZOO / ZOA).  Default.
      "voronoi" — Same repulsion/orientation/attraction logic but neighbour
                  selection uses the Voronoi/Delaunay graph instead of metric
                  radii.  Requires scipy.spatial (falls back gracefully).

    Social speed is an optional add-on (AgentType.social_speed > 0) that
    works on top of either interaction model.

    Cross-group schooling: types with no adversarial relationship are
    included in each other's social computation.
    Cross-group hunt/flee: applied as a separate per-agent pass after the
    social step.
    """

    def __init__(self, agent_types=None, seed=None):
        if agent_types is None:
            agent_types = [PREY, PREDATOR]
        self.agent_types: dict[str, object] = {t.name: t for t in agent_types}
        self.groups: dict[str, list[Boid]]  = {t.name: [] for t in agent_types}
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)
        self.env      = None   # set to an Environment instance to enable walls
        self._next_id = 0      # monotonically increasing; never reused

    # ------------------------------------------------------------------ #
    # Population management
    # ------------------------------------------------------------------ #
    def init_agents(self, type_name, n):
        atype  = self.agent_types[type_name]
        agents = self.groups[type_name]
        if self.env is not None:
            x0, x1, y0, y1 = self.env.spawn_bounds
        else:
            x0, x1 = 100, self.arena_width  - 100
            y0, y1 = 100, self.arena_height - 100
        new_agents = []
        for _ in range(n):
            new_agents.append(Boid(
                unique_id=self._next_id,
                x=self.random.uniform(x0, x1),
                y=self.random.uniform(y0, y1),
                dx=self.random.uniform(-3, 3),
                dy=self.random.uniform(-3, 3),
                agent_type=atype,
            ))
            self._next_id += 1
        agents.extend(new_agents)
        atype.assign_individual(new_agents, rng=self.rng)

    @property
    def arena_width(self):
        return self.env.width if self.env is not None else self.width

    @property
    def arena_height(self):
        return self.env.height if self.env is not None else self.height

    @property
    def all_agents(self):
        return [a for group in self.groups.values() for a in group]

    def clear(self):
        for agents in self.groups.values():
            agents.clear()

    # ------------------------------------------------------------------ #
    # Social interaction models
    #
    # Each returns a dict:
    #   des_x, des_y   — desired velocity direction (n_self,)
    #   n_repulsion    — number of repulsion-zone neighbours per agent
    #   ori_mask       — boolean (n_self, n_total) orientation neighbours
    #   n_ori          — count of orientation neighbours per agent
    #
    # Inputs are the shared arrays already computed in _step_group.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _social_zonal(vel, diff, dists, in_fov, n_self,
                      r_r_arr, r_o_arr, r_a_arr):
        """Couzin 2002 zone-based social forces."""
        zor = (dists <  r_r_arr[:, None])                                & in_fov
        zoo = (dists >= r_r_arr[:, None]) & (dists < r_o_arr[:, None])  & in_fov
        zoa = (dists >= r_o_arr[:, None]) & (dists < r_a_arr[:, None])  & in_fov

        n_zor = zor.sum(axis=1)
        n_zoo = zoo.sum(axis=1)
        n_zoa = zoa.sum(axis=1)

        safe_d      = np.where((dists == 0) | np.isinf(dists), 1.0, dists)[:, :, None]
        unit_away   =  diff / safe_d
        unit_toward = -unit_away

        rep_x = (unit_away[:, :, 0]  * zor).sum(axis=1)
        rep_y = (unit_away[:, :, 1]  * zor).sum(axis=1)

        h_all = np.arctan2(vel[:, 1], vel[:, 0])
        ori_x = (np.cos(h_all)[None, :] * zoo).sum(axis=1)
        ori_y = (np.sin(h_all)[None, :] * zoo).sum(axis=1)

        att_x = (unit_toward[:, :, 0] * zoa).sum(axis=1)
        att_y = (unit_toward[:, :, 1] * zoa).sum(axis=1)

        has_soc = (n_zoo > 0) | (n_zoa > 0)
        soc_x = np.where(n_zoo > 0, ori_x, 0.0) + np.where(n_zoa > 0, att_x, 0.0)
        soc_y = np.where(n_zoo > 0, ori_y, 0.0) + np.where(n_zoa > 0, att_y, 0.0)

        des_x = np.where(n_zor > 0, rep_x, np.where(has_soc, soc_x, vel[:n_self, 0]))
        des_y = np.where(n_zor > 0, rep_y, np.where(has_soc, soc_y, vel[:n_self, 1]))

        return dict(des_x=des_x, des_y=des_y,
                    n_repulsion=n_zor, ori_mask=zoo, n_ori=n_zoo)

    @staticmethod
    def _social_voronoi(vel, diff, dists, in_fov, n_self, pos,
                        r_r_arr, r_o_arr, r_a_arr):
        """
        Voronoi/Delaunay neighbour-based social forces.

        Neighbour selection: two agents are neighbours if they share a
        Delaunay edge (equivalent to the Voronoi diagram ridge).  This
        replaces the metric-radius membership test; the zone-radius
        parameters are reinterpreted as:
          r_r  — repulsion threshold  (too close → repel)
          r_o  — orientation distance (within this and not too close → align)
          r_a  — max interaction distance (Voronoi neighbours beyond this
                 are ignored entirely)

        Degenerate triangulations fall back to metric zones.
        """
        n_total = dists.shape[1]
        # Two or three agents are all neighbours. Degenerate larger groups
        # use metric zones rather than silently losing every social interaction.
        voronoi_nb = np.ones((n_self, n_total), dtype=bool)
        if n_total >= 3:
            try:
                tri = Delaunay(pos)
            except QhullError:
                return Model._social_zonal(vel, diff, dists, in_fov, n_self,
                                           r_r_arr, r_o_arr, r_a_arr)
            voronoi_nb[:] = False
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = simplex[a], simplex[b]
                        if i < n_self:
                            voronoi_nb[i, j] = True
                        if j < n_self:
                            voronoi_nb[j, i] = True

        # Apply max-interaction-distance cutoff + FOV + exclude self
        # r_a_arr used as max distance; within that apply the zone logic
        within_max = (dists < r_a_arr[:, None]) & in_fov
        nb = voronoi_nb & within_max

        # Zone membership among Voronoi neighbours
        zor = nb & (dists <  r_r_arr[:, None])
        zoo = nb & (dists >= r_r_arr[:, None]) & (dists < r_o_arr[:, None])
        zoa = nb & (dists >= r_o_arr[:, None])   # up to max distance (r_a cutoff above)

        n_zor = zor.sum(axis=1)
        n_zoo = zoo.sum(axis=1)
        n_zoa = zoa.sum(axis=1)

        safe_d      = np.where((dists == 0) | np.isinf(dists), 1.0, dists)[:, :, None]
        unit_away   =  diff / safe_d
        unit_toward = -unit_away

        rep_x = (unit_away[:, :, 0]  * zor).sum(axis=1)
        rep_y = (unit_away[:, :, 1]  * zor).sum(axis=1)

        h_all = np.arctan2(vel[:, 1], vel[:, 0])
        ori_x = (np.cos(h_all)[None, :] * zoo).sum(axis=1)
        ori_y = (np.sin(h_all)[None, :] * zoo).sum(axis=1)

        att_x = (unit_toward[:, :, 0] * zoa).sum(axis=1)
        att_y = (unit_toward[:, :, 1] * zoa).sum(axis=1)

        has_soc = (n_zoo > 0) | (n_zoa > 0)
        soc_x = np.where(n_zoo > 0, ori_x, 0.0) + np.where(n_zoa > 0, att_x, 0.0)
        soc_y = np.where(n_zoo > 0, ori_y, 0.0) + np.where(n_zoa > 0, att_y, 0.0)

        des_x = np.where(n_zor > 0, rep_x, np.where(has_soc, soc_x, vel[:n_self, 0]))
        des_y = np.where(n_zor > 0, rep_y, np.where(has_soc, soc_y, vel[:n_self, 1]))

        return dict(des_x=des_x, des_y=des_y,
                    n_repulsion=n_zor, ori_mask=zoo, n_ori=n_zoo)

    # ------------------------------------------------------------------ #
    # Per-group step — shared setup, model dispatch, shared post-processing
    # ------------------------------------------------------------------ #
    def _step_group(self, agents, atype, flee_targets, hunt_targets,
                    school_targets, desired_speed, fov_deg):
        n_self  = len(agents)
        n_total = n_self + len(school_targets)

        # Per-agent parameter arrays
        def _arr(param, default):
            if atype.has_distributions and any(param in a.individual for a in agents):
                return np.array([a.individual.get(param, default) for a in agents])
            return np.full(n_self, default)

        eff_speeds   = desired_speed * _arr("speed_scale", atype.speed_scale)
        r_r_arr      = _arr("r_r",      atype.r_r)
        r_o_arr      = _arr("r_o",      atype.r_o)
        r_a_arr      = _arr("r_a",      atype.r_a)
        max_turn_arr = _arr("max_turn", atype.max_turn)
        noise_arr    = _arr("noise",    atype.noise)

        # Combined position / velocity arrays (self agents first, then school)
        pos = np.empty((n_total, 2))
        vel = np.empty((n_total, 2))
        for i, a in enumerate(agents + school_targets):
            pos[i, 0] = a.x;  pos[i, 1] = a.y
            vel[i, 0] = a.dx; vel[i, 1] = a.dy

        # Pairwise distances and FOV mask
        diff    = pos[:n_self, None, :] - pos[None, :, :]   # (n_self, n_total, 2)
        bounded = self.env is not None and self.env.bounded
        if not bounded:
            diff[:, :, 0] -= np.round(diff[:, :, 0] / self.arena_width)  * self.arena_width
            diff[:, :, 1] -= np.round(diff[:, :, 1] / self.arena_height) * self.arena_height
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])      # (n_self, n_total)
        dists[np.arange(n_self), np.arange(n_self)] = np.inf

        fov_half = np.deg2rad(fov_deg) / 2
        headings = np.arctan2(vel[:n_self, 1], vel[:n_self, 0])
        angle_ij = np.arctan2(-diff[:, :, 1], -diff[:, :, 0])
        rel_ang  = (angle_ij - headings[:, None] + np.pi) % (2 * np.pi) - np.pi
        in_fov   = np.abs(rel_ang) <= fov_half

        if self.env is not None:
            los = self.env.los_mask(pos[:n_self], pos)
            if los is not None:
                in_fov = in_fov & los

        # ── Social direction (model-specific) ──────────────────────────
        imodel = getattr(atype, "interaction_model", "zonal")
        if imodel == "zonal":
            soc = self._social_zonal(vel, diff, dists, in_fov, n_self,
                                     r_r_arr, r_o_arr, r_a_arr)
        elif imodel == "voronoi":
            soc = self._social_voronoi(vel, diff, dists, in_fov, n_self, pos,
                                       r_r_arr, r_o_arr, r_a_arr)
        else:
            raise ValueError(f"Unknown interaction_model {imodel!r}. "
                             f"Choose 'zonal' or 'voronoi'.")

        des_x = soc["des_x"]
        des_y = soc["des_y"]

        # ── Noise + heading update (same for all models) ────────────────
        des_h = np.arctan2(des_y, des_x)
        noise = self.rng.normal(0, noise_arr)
        dh    = (des_h - headings + np.pi) % (2 * np.pi) - np.pi
        new_h = headings + np.clip(dh + noise, -max_turn_arr, max_turn_arr)

        # ── Social speed add-on (same for all models) ───────────────────
        # social_speed=0 → strict fixed speed per agent
        # social_speed>0 → ZOO/Voronoi-orientation speed matching + ZOR escape boost
        alpha = atype.social_speed
        if alpha > 0:
            n_repulsion = soc["n_repulsion"]
            ori_mask    = soc["ori_mask"]
            n_ori       = soc["n_ori"]

            cur = np.hypot(vel[:n_self, 0], vel[:n_self, 1])
            cur = np.where(cur < 1e-10, eff_speeds, cur)

            speeds_all  = np.hypot(vel[:, 0], vel[:, 1])
            avg_ori_spd = (speeds_all[None, :] * ori_mask).sum(axis=1) / np.maximum(n_ori, 1)
            ori_target  = np.where(n_ori > 0, avg_ori_spd, eff_speeds)

            crowd      = np.clip(n_repulsion / 3.0, 0.0, 1.0)
            rep_target = eff_speeds * (1.0 + alpha * crowd)

            target = np.where(n_repulsion > 0, rep_target, ori_target)
            spd    = cur + alpha * 0.15 * (target - cur)
            vel[:n_self, 0] = spd * np.cos(new_h)
            vel[:n_self, 1] = spd * np.sin(new_h)
        else:
            vel[:n_self, 0] = eff_speeds * np.cos(new_h)
            vel[:n_self, 1] = eff_speeds * np.sin(new_h)

        # ── Cross-group flee ─────────────────────────────────────────────
        if flee_targets:
            for i, a in enumerate(agents):
                a.dx = float(vel[i, 0]); a.dy = float(vel[i, 1])
                fd = a.individual.get("flee_dist", atype.flee_dist)
                detect_preds(a, flee_targets, fd)
                flee(a, flee_targets, fd, float(eff_speeds[i]))
                vel[i, 0] = a.dx; vel[i, 1] = a.dy

        # ── Cross-group hunt ─────────────────────────────────────────────
        if hunt_targets:
            for i, a in enumerate(agents):
                a.dx = float(vel[i, 0]); a.dy = float(vel[i, 1])
                hunt(a, hunt_targets, float(r_a_arr[i]))
                vel[i, 0] = a.dx; vel[i, 1] = a.dy

        # ── Wall forces ──────────────────────────────────────────────────
        if self.env is not None:
            self.env.apply_forces(pos[:n_self], vel[:n_self],
                                  self.arena_width, self.arena_height)

        # ── Speed renormalisation ────────────────────────────────────────
        sp = np.hypot(vel[:n_self, 0], vel[:n_self, 1])
        sp = np.maximum(sp, 1e-10)
        if alpha > 0:
            cap   = eff_speeds * (1.0 + alpha * 1.5)
            scale = np.where(sp > cap, cap / sp, 1.0)
        else:
            scale = eff_speeds / sp
        vel[:n_self, 0] *= scale
        vel[:n_self, 1] *= scale

        # ── Position update ──────────────────────────────────────────────
        new_x = pos[:n_self, 0] + vel[:n_self, 0]
        new_y = pos[:n_self, 1] + vel[:n_self, 1]
        if self.env is not None and self.env.bounded:
            pos[:n_self, 0] = np.clip(new_x, 0, self.arena_width  - 1)
            pos[:n_self, 1] = np.clip(new_y, 0, self.arena_height - 1)
        else:
            pos[:n_self, 0] = new_x % self.arena_width
            pos[:n_self, 1] = new_y % self.arena_height
        for i, a in enumerate(agents):
            a.dx = float(vel[i, 0]); a.dy = float(vel[i, 1])
            a.x  = float(pos[i, 0]); a.y  = float(pos[i, 1])

    # ------------------------------------------------------------------ #
    # Main step
    # ------------------------------------------------------------------ #
    def step(self, catch_dist, desired_speed, fov_deg=270):
        import copy
        original = self.groups
        self.groups = {name: [copy.copy(a) for a in agents]
                       for name, agents in original.items()}
        starting = self.groups
        survivors = {}
        for type_name in sorted(starting):
            hunters = [
                a
                for tn, t in self.agent_types.items()
                if type_name in t.hunts
                for a in self.groups.get(tn, [])
            ]
            survivors[type_name] = [
                a for a in starting[type_name]
                if not is_eaten(a, hunters, catch_dist)
            ]
        self.groups = survivors
        snapshot = {name: [copy.copy(a) for a in agents]
                    for name, agents in survivors.items()}

        for type_name in sorted(self.groups):
            agents = self.groups[type_name]
            if not agents:
                continue
            atype = self.agent_types[type_name]

            flee_targets = [a for fn in sorted(atype.flees) for a in snapshot.get(fn, [])]
            hunt_targets = [a for hn in sorted(atype.hunts) for a in snapshot.get(hn, [])]

            school_targets = [
                a
                for other_name, other_agents in sorted(snapshot.items())
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

        # Commit all next states together, retaining public agent object identity.
        for name, agents in self.groups.items():
            by_id = {a.unique_id: a for a in original[name]}
            for index, updated in enumerate(agents):
                agent = by_id[updated.unique_id]
                agent.x, agent.y = updated.x, updated.y
                agent.dx, agent.dy, agent.state = updated.dx, updated.dy, updated.state
                agents[index] = agent
