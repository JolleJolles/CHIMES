import math
import numpy as np


# ---------------------------------------------------------------------------
# Parameter name aliases — friendly names used in set_distribution()
# map to the internal AgentType attribute names.
#
#   "speed"    → speed_scale   (multiplier on the global speed slider)
#   "zor"      → r_r           (zone of repulsion radius)
#   "zoo"      → r_o           (zone of orientation radius)
#   "zoa"      → r_a           (zone of attraction radius)
#   "detect"   → flee_dist     (threat detection range)
#   "socspeed" → social_speed  (longitudinal social-force strength)
#   "maxturn"  → max_turn      (maximum turn per step, radians)
#   "noise"    → noise         (already the right name)
# ---------------------------------------------------------------------------

_PARAM_ALIASES = {
    "speed":     "speed_scale",
    "zor":       "r_r",
    "zoo":       "r_o",
    "zoa":       "r_a",
    "detect":    "flee_dist",
    "socspeed":  "social_speed",
    "soc_speed": "social_speed",
    "maxturn":   "max_turn",
}


def _resolve_param(name):
    """Return the internal AgentType attribute name for a user-provided alias."""
    name = name.lower()
    return _PARAM_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Colour mapping helper
# ---------------------------------------------------------------------------

def _values_to_colors(values, colormap_name):
    """Map a 1-D numpy array of floats to a list of '#rrggbb' hex strings.

    Uses matplotlib if available; falls back to a blue→red gradient.
    """
    values = np.asarray(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    norm = (values - vmin) / (vmax - vmin) if vmax > vmin else np.full(len(values), 0.5)

    try:
        from matplotlib import colormaps
        cmap = colormaps[colormap_name]
        rgba = cmap(norm)
        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in rgba
        ]
    except Exception:
        # Fallback: blue (#56b1fc) → red (#fc6056)
        r0, g0, b0 = 0x56, 0xb1, 0xfc
        r1, g1, b1 = 0xfc, 0x60, 0x56
        return [
            "#{:02x}{:02x}{:02x}".format(
                int(r0 + t * (r1 - r0)),
                int(g0 + t * (g1 - g0)),
                int(b0 + t * (b1 - b0)),
            )
            for t in norm.tolist()
        ]


# ---------------------------------------------------------------------------
# AgentType
# ---------------------------------------------------------------------------

class AgentType:
    """
    Configuration for one class of agents.

    All mutable attributes can be updated at runtime — GUI sliders write
    to these directly so changes take effect immediately.

    Individual variation
    --------------------
    Call set_distribution() to register a sampling distribution for supported
    movement parameters.  When agents are initialised (or reset), each one
    receives its own sampled value stored in Boid.individual.  Agents can
    optionally be coloured by their individual value on a colormap.

    Parameters
    ----------
    name, color, size, n_default   — display / spawning
    r_r, r_o, r_a                  — Couzin zone radii (repulsion / orientation / attraction)
    flee_dist                      — detection range for threats
    max_turn                       — max heading change per step (radians)
    speed_scale                    — multiplier on global speed slider
    social_speed                   — strength of longitudinal social forces (0–1)
    preferred_distance             — spacing where social distance force changes sign
    speed_relaxation               — preferred-speed recovery rate per step
    speed_noise                    — temporal noise amplitude / preferred speed
    max_speed_scale                — upper speed limit / preferred speed
    noise                          — Gaussian heading-noise standard deviation (radians)
    pursues, flees, captures       — directed cross-type relationships; capture is opt-in
    """

    def __init__(self, name, *,
                 color="#56b1fc",
                 size=15,
                 n_default=100,
                 r_r=10.0,
                 r_o=30.0,
                 r_a=75.0,
                 flee_dist=100.0,
                 visual_range_multiplier=1.0,
                 max_turn=math.pi * 0.2,
                 speed_scale=1.0,
                 social_speed=0.0,
                 preferred_distance=30.0,
                 speed_relaxation=0.15,
                 speed_noise=0.0,
                 max_speed_scale=3.0,
                 noise=0.1,
                 heading_noise_memory=0.0,
                 interaction_model="zonal",
                 pursues=(),
                 flees=(),
                 captures=()):
        self.name              = name
        self.color             = color
        self.size              = size
        self.n_default         = n_default
        self.r_r               = float(r_r)
        self.r_o               = float(r_o)
        self.r_a               = float(r_a)
        self.flee_dist         = float(flee_dist)
        self.visual_range_multiplier = float(visual_range_multiplier)
        self.max_turn          = float(max_turn)
        self.speed_scale       = float(speed_scale)
        self.social_speed      = float(social_speed)
        self.preferred_distance = float(preferred_distance)
        self.speed_relaxation   = float(speed_relaxation)
        self.speed_noise        = float(speed_noise)
        self.max_speed_scale    = float(max_speed_scale)
        self.noise             = float(noise)
        self.heading_noise_memory = float(heading_noise_memory)
        self.interaction_model = interaction_model   # "zonal" | "voronoi"
        for key, value in (('pursues', pursues), ('flees', flees), ('captures', captures)):
            if isinstance(value, str):
                raise ValueError(f'{key} must be a collection of agent-type names, not a string')
            setattr(self, key, set(value))

        # Individual variation
        self._distributions: dict = {}   # param → spec dict
        self.color_by:  str | None = None
        self.colormap:  str        = "viridis"

    def validate(self):
        """Reject invalid physical parameters before agents are initialised."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Agent types need a nonempty name")
        if isinstance(self.n_default, bool) or not isinstance(self.n_default, int) or self.n_default < 0:
            raise ValueError("n_default must be a nonnegative integer")
        for param in ("size", "r_r", "r_o", "r_a", "flee_dist", "visual_range_multiplier", "speed_scale", "noise", "max_turn", "social_speed",
                      "preferred_distance", "speed_relaxation", "speed_noise", "max_speed_scale", "heading_noise_memory"):
            value = getattr(self, param)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{self.name}: {param} must be finite and nonnegative")
        if self.preferred_distance <= 0 or self.max_speed_scale < 1:
            raise ValueError("preferred_distance must be positive and max_speed_scale must be >= 1")
        if not self.r_r <= self.r_o <= self.r_a:
            raise ValueError(f"{self.name}: require r_r <= r_o <= r_a")
        if not 0 <= self.social_speed <= 1 or not 0 <= self.max_turn <= math.pi:
            raise ValueError("social_speed must be in [0, 1] and max_turn in [0, pi]")
        if self.interaction_model not in {"zonal", "voronoi"}:
            raise ValueError("interaction_model must be 'zonal' or 'voronoi'")

        for key in ('pursues', 'flees', 'captures'):
            if any(not isinstance(name, str) or not name.strip() for name in getattr(self, key)):
                raise ValueError(f'{self.name}: {key} must contain nonempty agent-type names')
            if self.name in getattr(self, key):
                raise ValueError(f'{self.name}: {key} cannot target its own type')
        if self.pursues & self.flees:
            raise ValueError(f'{self.name}: cannot pursue and flee the same type')

    # ------------------------------------------------------------------ #
    # Distribution API
    # ------------------------------------------------------------------ #

    def clone(self, name=None):
        """Return an independent copy of this AgentType.

        Use this when you want to start from an existing preset and change
        a few things, without modifying the original:

            fast = BOID.clone("fast")
            fast.speed_scale = 1.5
            fast.set_distribution("speed_scale", dist="normal")
        """
        import copy as _copy
        c = _copy.deepcopy(self)
        if name is not None:
            c.name = name
        return c

    def to_dict(self):
        """Return a JSON-compatible configuration; angles are in radians."""
        import copy
        fields = ("color", "size", "n_default", "r_r", "r_o", "r_a", "flee_dist", "visual_range_multiplier",
                  "max_turn", "speed_scale", "social_speed", "preferred_distance",
                  "speed_relaxation", "speed_noise", "max_speed_scale", "noise", "heading_noise_memory", "interaction_model")
        return {**{key: getattr(self, key) for key in fields},
                "pursues": sorted(self.pursues), "flees": sorted(self.flees), "captures": sorted(self.captures),
                "distributions": copy.deepcopy(self._distributions),
                "color_by": self.color_by, "colormap": self.colormap}

    @classmethod
    def from_dict(cls, name, values):
        """Read Python parameters or a saved GUI agent configuration."""
        import copy
        values = copy.deepcopy(values)
        # Old hunting implied both steering and removal. Loading now retains
        # steering only; capture always requires an explicit captures list.
        if 'hunts' in values:
            legacy = values.pop('hunts')
            values.setdefault('pursues', legacy)
        for alias, key in (("n", "n_default"), ("repulsion", "r_r"),
                           ("orientation", "r_o"), ("attraction", "r_a")):
            if alias in values:
                values[key] = values.pop(alias)
        for key in ("max_turn", "noise"):
            if key + "_deg" in values:
                values[key] = math.radians(values.pop(key + "_deg"))
        distributions = values.pop("distributions", {})
        color_by = values.pop("color_by", None)
        colormap = values.pop("colormap", "viridis")
        agent_type = cls(name, **values)
        for param, spec in distributions.items():
            spec = dict(spec)
            dist = spec.pop("_dist", spec.pop("dist", "normal"))
            clip = spec.pop("_clip", spec.pop("clip", None))
            truncate = spec.pop("_truncate", spec.pop("truncate", False))
            agent_type.set_distribution(param, dist=dist, clip=clip, truncate=truncate, color_by=False, **spec)
        agent_type.color_by = _resolve_param(color_by) if color_by is not None else None
        agent_type.colormap = colormap
        agent_type._color_label = color_by
        return agent_type

    def set_distribution(self, param, dist="normal", clip=None,
                         color_by=None, colormap="viridis", truncate=False, **kwargs):
        """Register a sampling distribution for one parameter.

        Friendly name aliases (case-insensitive) are accepted for param:
          "speed"    → speed multiplier on the global speed slider
          "zor"      → repulsion zone radius
          "zoo"      → orientation zone radius
          "zoa"      → attraction zone radius
          "detect"   → threat detection range
          "socspeed" → longitudinal social-force strength
          "maxturn"  → maximum turn per step
          "noise"    → heading noise (already the right name)

        All distribution kwargs are optional — defaults are derived from the
        current type-level value of the parameter (mean = current value,
        spread = 20 % of that value).

        Parameters
        ----------
        param    : parameter to vary (friendly name or internal attribute)
        dist     : "normal"   — bell curve;   kwargs: mean, std
                   "uniform"  — flat range;   kwargs: low, high
                   "lognormal"— right-skewed; kwargs: mean, std (log-space)
                   "bimodal"  — two peaks;    kwargs: means=[m1,m2],
                                              stds=[s1,s2], weights=[w1,w2]
        clip     : optional (min, max) bounds; physical bounds always apply
        truncate : for normal distributions, sample inside bounds instead of clipping
        color_by : True  → colour agents by this param (overrides any prior)
                   None  → auto-enable if this is the first distribution added
                   False → never colour by this param
        colormap : matplotlib colormap name ("viridis", "plasma", "RdYlGn", …)

        Minimal usage (all defaults — no kwargs needed)
        -----------------------------------------------
        fish.set_distribution("speed")          # normal centred on speed_scale
        fish.set_distribution("zoa")            # normal centred on r_a
        fish.set_distribution("zoa", dist="bimodal")   # two peaks at 0.6× and 1.4×
        fish.set_distribution("detect", dist="uniform") # flat range 0.5×–1.5×

        Full control
        ------------
        fish.set_distribution("speed",
            dist="normal", mean=1.0, std=0.3, clip=(0.2, 3.0),
            color_by=True, colormap="plasma")
        fish.set_distribution("zoa",
            dist="bimodal", means=[40, 120], stds=[10, 15], weights=[0.6, 0.4])
        """
        internal = _resolve_param(param)
        supported = {"speed_scale", "r_r", "r_o", "r_a", "flee_dist", "max_turn", "noise", "social_speed",
                     "preferred_distance", "speed_relaxation", "speed_noise", "max_speed_scale"}
        if internal not in supported:
            raise ValueError(f"Unsupported individual parameter {param!r}; choose {sorted(supported)}")
        allowed = {"normal": {"mean", "std"}, "uniform": {"low", "high"},
                   "lognormal": {"mean", "std"}, "bimodal": {"means", "stds", "weights"}}
        if dist not in allowed:
            raise ValueError(f"Unknown distribution {dist!r}")
        if set(kwargs) - allowed[dist]:
            raise ValueError(f"Unknown {dist} parameters: {sorted(set(kwargs) - allowed[dist])}")
        if clip is not None:
            if len(clip) != 2:
                raise ValueError("clip must contain (minimum, maximum)")
            lower, upper = (-np.inf if clip[0] is None else clip[0], np.inf if clip[1] is None else clip[1])
            if np.isnan(lower) or np.isnan(upper) or lower > upper:
                raise ValueError("clip must contain ordered bounds; None means unbounded")
        spec = dict(kwargs)
        spec["_dist"] = dist
        spec["_clip"] = clip
        if truncate:
            if dist != "normal":
                raise ValueError("truncate is supported only for normal distributions")
            spec["_truncate"] = True
        self._distributions[internal] = spec

        # Coloring: auto-enable for the FIRST distribution added (color_by=None),
        # force-enable if color_by=True, never enable if color_by=False.
        if color_by is not False:
            if color_by is True or self.color_by is None:
                self.color_by    = internal
                self._color_label = param      # keep user's name for display
                self.colormap    = colormap

    def _sample(self, n, rng=None, parameters=None):
        """Sample all registered distributions for n agents.

        When a distribution-specific kwarg is omitted, a sensible default is
        derived from the current type-level value of the parameter:
          normal    — mean = current value,  std  = 20 % of mean
          uniform   — low  = 50 % of value,  high = 150 % of value
          lognormal — mean = ln(value),       std  = 0.3  (log-space)
          bimodal   — peaks at 60 % and 140 % of value,
                      each with std = 10 % of value, equal weights

        Returns dict of {param: np.ndarray(n, dtype=float)}.
        """
        rng = rng if rng is not None else np.random.default_rng()
        out = {}
        for param, spec in self._distributions.items():
            if parameters is not None and param not in parameters:
                continue
            d = spec["_dist"]
            # Current type-level value used as the natural centre/scale
            val    = float(getattr(self, param, 1.0))
            spread = max(abs(val) * 0.2, 1e-3)   # 20 % or tiny floor

            if d == "normal":
                mean = spec.get("mean", val)
                std  = spec.get("std",  spread)
                if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
                    raise ValueError("Normal mean must be finite and SD finite and nonnegative")
                if spec.get("_truncate"):
                    from scipy.stats import truncnorm
                    lower, upper = spec.get("_clip") or (-np.inf, np.inf)
                    lower = -np.inf if lower is None else lower
                    upper = np.inf if upper is None else upper
                    lower = max(lower, {"preferred_distance": 1e-6, "max_speed_scale": 1.}.get(param, 0.))
                    upper = min(upper, {"social_speed": 1., "max_turn": math.pi}.get(param, np.inf))
                    if lower > upper or (std == 0 and not lower <= mean <= upper):
                        raise ValueError("Normal bounds must overlap physical limits and contain a constant value")
                    if lower == upper or std == 0:
                        v = np.full(n, lower if lower == upper else mean)
                    else:
                        v = truncnorm.rvs((lower - mean) / std, (upper - mean) / std,
                                          loc=mean, scale=std, size=n, random_state=rng)
                else:
                    v = rng.normal(mean, std, n)

            elif d == "uniform":
                low  = spec.get("low",  val * 0.5)
                high = spec.get("high", val * 1.5)
                v = rng.uniform(low, high, n)

            elif d == "lognormal":
                # mean/std are in log-space
                mean = spec.get("mean", math.log(max(val, 1e-9)))
                std  = spec.get("std",  0.3)
                v = rng.lognormal(mean, std, n)

            elif d == "bimodal":
                means   = spec.get("means",   [val * 0.6, val * 1.4])
                stds    = spec.get("stds",    [spread * 0.5, spread * 0.5])
                weights = np.asarray(spec.get("weights", [0.5, 0.5]), dtype=float)
                weights /= weights.sum()
                idx = rng.choice(len(weights), size=n, p=weights)
                v = np.array([rng.normal(means[k], stds[k]) for k in idx])

            else:
                raise ValueError(f"Unknown distribution {d!r} for param {param!r}")

            clip = spec.get("_clip")
            if clip is not None and any(value is not None for value in clip):
                v = np.clip(v, clip[0], clip[1])
            upper = {"social_speed": 1., "max_turn": math.pi}.get(param, np.inf)
            lower = {"preferred_distance": 1e-6, "max_speed_scale": 1.}.get(param, 0.)
            v = np.clip(v, lower, upper)
            if not np.isfinite(v).all():
                raise ValueError(f"Distribution for {param} produced nonfinite values")
            out[param] = v.astype(float)
        return out

    def constrain_zones(self, maximum, agents=()):
        """Keep GUI radii, distributions and current individuals within detection.

        Tightening limits clips affected traits; it does not resample agents or
        alter their positions. Expanding limits does not restore clipped values.
        """
        self.r_a = min(self.r_a, maximum)
        self.r_o = min(self.r_o, self.r_a)
        self.r_r = min(self.r_r, self.r_o)
        bounds = dict(r_r=(0., self.r_o), r_o=(self.r_r, self.r_a), r_a=(self.r_o, maximum))
        for key, (lower, upper) in bounds.items():
            spec = self._distributions.get(key)
            if spec is None:
                continue
            old_lower, old_upper = spec.get('_clip') or (None, None)
            low = max(lower, lower if old_lower is None else min(old_lower, upper))
            high = min(upper, upper if old_upper is None else max(old_upper, low))
            spec['_clip'] = [low, high]
            if spec['_dist'] == 'normal':
                spec['mean'] = min(high, max(low, getattr(self, key)))
            elif spec['_dist'] == 'uniform':
                spec['low'] = min(high, max(low, spec.get('low', lower)))
                spec['high'] = min(high, max(spec['low'], spec.get('high', upper)))
        for agent in agents:
            ra = min(maximum, agent.individual.get('r_a', self.r_a))
            ro = min(ra, agent.individual.get('r_o', self.r_o))
            rr = min(ro, agent.individual.get('r_r', self.r_r))
            for key, value in (('r_r', rr), ('r_o', ro), ('r_a', ra)):
                if key in agent.individual or value != getattr(self, key):
                    agent.individual[key] = float(value)
        self.update_colors(agents) if agents else None

    def _ordered_samples(self, n, rng=None, parameters=None, agents=None):
        """Condition sampled zone radii on valid ordering without clipping or sorting.

        Retry only varied radii of rejected individuals; keep other traits and
        unedited individual radii unchanged. An incompatible configuration fails
        before assigning values to any agents.
        """
        rng = rng if rng is not None else np.random.default_rng()
        samples = self._sample(n, rng=rng, parameters=parameters)
        keys = ("r_r", "r_o", "r_a")
        varied = set(keys) & samples.keys()
        for attempt in range(129):
            radii = []
            for key in keys:
                if key in samples:
                    values = samples[key]
                elif agents is not None and parameters is not None and key not in parameters:
                    values = np.array([a.individual.get(key, getattr(self, key)) for a in agents])
                else:
                    values = np.full(n, getattr(self, key))
                radii.append(values)
            invalid = (radii[0] > radii[1]) | (radii[1] > radii[2])
            if not invalid.any():
                return samples
            if not varied or attempt == 128:
                raise ValueError("Cannot sample ordered zone radii: repulsion ≤ orientation ≤ attraction. "
                                 "Adjust the radius distribution bounds or neighbouring radii; "
                                 "no valid combination was found within 128 retries.")
            replacement = self._sample(int(invalid.sum()), rng=rng, parameters=varied)
            for key in varied:
                samples[key][invalid] = replacement[key]

    def assign_individual(self, agents, rng=None):
        """Sample distributions and write individual values + colours into agents.

        Called automatically by Model.init_agents() on every newly created Boid.
        Also pre-generates the colorbar gradient so the HUD can draw it cheaply.
        """
        if not self._distributions or not agents:
            return
        n = len(agents)
        samples = self._ordered_samples(n, rng=rng)

        for i, a in enumerate(agents):
            for param, vals in samples.items():
                a.individual[param] = float(vals[i])
        self.update_colors(agents)

    def apply_variation(self, agents, param, rng=None):
        """Apply one trait to existing agents without moving them or resampling other traits.

        Removing its distribution first makes that trait fixed at the type value.
        Samples are conditioned on valid zone ordering before individual values change.
        """
        params = {_resolve_param(p) for p in ([param] if isinstance(param, str) else param)}
        if not agents:
            return
        self.validate()
        samples = self._ordered_samples(len(agents), rng=rng, parameters=params, agents=agents)
        values = {key: samples.get(key, np.full(len(agents), getattr(self, key))) for key in params}
        for i, a in enumerate(agents):
            for key in params:
                if key in samples:
                    a.individual[key] = float(values[key][i])
                else:
                    a.individual.pop(key, None)
        self.update_colors(agents)

    def update_colors(self, agents):
        """Colour by an effective trait, including a uniform colour for fixed traits."""
        if not self.color_by or not agents:
            for a in agents:
                a.color = None
            return
        values = np.array([a.individual.get(self.color_by, getattr(self, self.color_by)) for a in agents])
        self._dist_min, self._dist_max = float(values.min()), float(values.max())
        self._colorbar = _values_to_colors(np.linspace(self._dist_min, self._dist_max, 48), self.colormap)
        for a, colour in zip(agents, _values_to_colors(values, self.colormap)):
            a.color = colour

    @property
    def has_distributions(self):
        return bool(self._distributions)

    @property
    def has_colorbar(self):
        return self.color_by is not None and hasattr(self, "_colorbar")


# ---------------------------------------------------------------------------
# Boid
# ---------------------------------------------------------------------------

class Boid:
    """Per-agent state.  Behaviour comes from its AgentType + Model.step()."""

    __slots__ = ("unique_id", "x", "y", "dx", "dy", "type", "state", "individual", "color", "heading", "heading_error")

    def __init__(self, unique_id, x, y, dx, dy, agent_type):
        self.unique_id  = unique_id
        self.x          = x
        self.y          = y
        self.dx         = dx
        self.dy         = dy
        self.heading    = math.atan2(dy, dx)
        self.heading_error = None
        self.type       = agent_type
        self.state      = "calm"
        self.individual = {}    # {param: value} — per-agent overrides
        self.color      = None  # set to a hex string when color_by is active

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

BOID = AgentType(
    "boid",
    color="#56b1fc",
    size=12,
    n_default=100,
    r_r=10.0, r_o=30.0, r_a=75.0,
    flee_dist=100.0,
    max_turn=math.pi * 0.2,
    speed_scale=1.0,
    noise=0.09,
)



def validate_relationships(agent_types):
    """Reject misspelled or missing target types before starting a simulation."""
    names = {t.name for t in agent_types}
    for agent_type in agent_types:
        agent_type.validate()
        for key in ('pursues', 'flees', 'captures'):
            unknown = getattr(agent_type, key) - names
            if unknown:
                raise ValueError(f"{agent_type.name}: unknown {key} targets: {sorted(unknown)}")
