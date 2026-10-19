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
#   "socspeed" → social_speed  (speed-matching strength)
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
    return _PARAM_ALIASES.get(name, _PARAM_ALIASES.get(name.lower(), name))


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
    Call set_distribution() to register a sampling distribution for any
    numeric parameter.  When agents are initialised (or reset), each one
    receives its own sampled value stored in Boid.individual.  Agents can
    optionally be coloured by their individual value on a colormap.

    Parameters
    ----------
    name, color, size, n_default   — display / spawning
    r_r, r_o, r_a                  — Couzin zone radii (repulsion / orientation / attraction)
    flee_dist                      — detection range for threats
    max_turn                       — max heading change per step (radians)
    speed_scale                    — multiplier on global speed slider
    social_speed                   — 0 = fixed speed, 1 = full speed matching
    noise                          — per-step heading noise η (radians)
    hunts, flees                   — cross-type predation relationships
    """

    def __init__(self, name, *,
                 color="#56b1fc",
                 size=15,
                 n_default=100,
                 r_r=10.0,
                 r_o=30.0,
                 r_a=75.0,
                 flee_dist=100.0,
                 max_turn=math.pi * 0.2,
                 speed_scale=1.0,
                 social_speed=0.0,
                 noise=0.1,
                 interaction_model="zonal",
                 hunts=(),
                 flees=()):
        self.name              = name
        self.color             = color
        self.size              = size
        self.n_default         = n_default
        self.r_r               = float(r_r)
        self.r_o               = float(r_o)
        self.r_a               = float(r_a)
        self.flee_dist         = float(flee_dist)
        self.max_turn          = float(max_turn)
        self.speed_scale       = float(speed_scale)
        self.social_speed      = float(social_speed)
        self.noise             = float(noise)
        self.interaction_model = interaction_model   # "zonal" | "voronoi"
        self.hunts             = set(hunts)
        self.flees             = set(flees)

        # Individual variation
        self._distributions: dict = {}   # param → spec dict
        self.color_by:  str | None = None
        self.colormap:  str        = "viridis"

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
        fields = ("color", "size", "n_default", "r_r", "r_o", "r_a", "flee_dist",
                  "max_turn", "speed_scale", "social_speed", "noise", "interaction_model")
        return {**{key: getattr(self, key) for key in fields},
                "hunts": sorted(self.hunts), "flees": sorted(self.flees),
                "distributions": copy.deepcopy(self._distributions),
                "color_by": self.color_by, "colormap": self.colormap}

    @classmethod
    def from_dict(cls, name, values):
        """Read Python parameters or a saved GUI agent configuration."""
        import copy
        values = copy.deepcopy(values)
        if "n" in values:
            values["n_default"] = values.pop("n")
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
            agent_type.set_distribution(param, dist=dist, clip=clip, color_by=False, **spec)
        agent_type.color_by = color_by
        agent_type.colormap = colormap
        agent_type._color_label = color_by
        return agent_type

    def set_distribution(self, param, dist="normal", clip=None,
                         color_by=None, colormap="viridis", **kwargs):
        """Register a sampling distribution for one parameter.

        Friendly name aliases (case-insensitive) are accepted for param:
          "speed"    → speed multiplier on the global speed slider
          "zor"      → repulsion zone radius
          "zoo"      → orientation zone radius
          "zoa"      → attraction zone radius
          "detect"   → threat detection range
          "socspeed" → social speed matching strength
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
        clip     : (min, max) hard bounds after sampling; None = no clipping
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
        internal = _resolve_param(param)      # map alias → attribute name
        spec = dict(kwargs)
        spec["_dist"] = dist
        spec["_clip"] = clip
        self._distributions[internal] = spec

        # Coloring: auto-enable for the FIRST distribution added (color_by=None),
        # force-enable if color_by=True, never enable if color_by=False.
        if color_by is not False:
            if color_by is True or self.color_by is None:
                self.color_by    = internal
                self._color_label = param      # keep user's name for display
                self.colormap    = colormap

    def _sample(self, n, rng=None):
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
            d = spec["_dist"]
            # Current type-level value used as the natural centre/scale
            val    = float(getattr(self, param, 1.0))
            spread = max(abs(val) * 0.2, 1e-3)   # 20 % or tiny floor

            if d == "normal":
                mean = spec.get("mean", val)
                std  = spec.get("std",  spread)
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
            if clip is not None:
                v = np.clip(v, clip[0], clip[1])
            out[param] = v.astype(float)
        return out

    def assign_individual(self, agents, rng=None):
        """Sample distributions and write individual values + colours into agents.

        Called automatically by Model.init_agents() on every newly created Boid.
        Also pre-generates the colorbar gradient so the HUD can draw it cheaply.
        """
        if not self._distributions or not agents:
            return
        n = len(agents)
        samples = self._sample(n, rng=rng)

        colors = None
        if self.color_by and self.color_by in samples:
            cb_vals = samples[self.color_by]
            colors  = _values_to_colors(cb_vals, self.colormap)
            # Store range + pre-built gradient (48 steps) for the HUD colorbar
            self._dist_min = float(cb_vals.min())
            self._dist_max = float(cb_vals.max())
            self._colorbar = _values_to_colors(np.linspace(0.0, 1.0, 48), self.colormap)

        for i, a in enumerate(agents):
            for param, vals in samples.items():
                a.individual[param] = float(vals[i])
            if colors is not None:
                a.color = colors[i]

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

    __slots__ = ("unique_id", "x", "y", "dx", "dy", "type", "state", "individual", "color")

    def __init__(self, unique_id, x, y, dx, dy, agent_type):
        self.unique_id  = unique_id
        self.x          = x
        self.y          = y
        self.dx         = dx
        self.dy         = dy
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

PREY = AgentType(
    "prey",
    color="#56b1fc",
    size=15,
    n_default=100,
    r_r=10.0, r_o=30.0, r_a=75.0,
    flee_dist=100.0,
    max_turn=math.pi * 0.2,
    speed_scale=1.0,
    noise=0.09,
    flees=("predator",),
)

PREDATOR = AgentType(
    "predator",
    color="#fc6056",
    size=20,
    n_default=5,
    r_r=10.0, r_o=50.0, r_a=150.0,
    flee_dist=100.0,
    max_turn=math.pi * 0.15,
    speed_scale=1.5,
    noise=0.52,   # ~0.3 uniform half-range → 0.3*√3 ≈ 0.52 Gaussian σ
    hunts=("prey",),
)
