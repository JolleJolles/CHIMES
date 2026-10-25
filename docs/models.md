# Movement models

Choose the model in Controls or the dataset setup GUI. The same identifiers work
with `chimesim.settings(model=...)`, saved JSON files and simulation conditions.
All implementations here operate in two dimensions, with one movement update per
model step. They are adaptations with explicit rules, not validated reproductions
of every figure in the cited papers.

| Identifier | Model and relevant controls |
| --- | --- |
| `couzin2002` | **Zonal - Couzin et al. 2002 - 2D.** Repulsion radius, orientation outer radius, attraction outer radius, speed, field of view, maximum turn and Gaussian heading-noise SD. Neighbours occupy disjoint annuli; any repulsion overrides orientation and attraction for that update. |
| `zonal_overlap` | **Zonal with overlapping alignment and attraction.** Adds an attraction inner radius independent of the orientation outer radius. The same neighbour can contribute to both responses. Setting the attraction inner radius equal to the orientation outer radius recovers disjoint zones. This is a CHIMES extension, not a separately attributed published model. |
| `couzin2005` | **Zonal - Couzin et al. 2005 - 2D (uninformed).** Repulsion radius and one shared alignment/attraction radius. Outside avoidance, neighbours contribute their unit headings and directions towards their positions, with the focal heading included in the orientation sum. No informed individuals or preferred travel directions are implemented yet. |
| `vicsek` | **Vicsek et al. 1995 - 2D.** One interaction radius, fixed speed and uniform angular noise. The noise value is the *full width* of the uniform distribution, so `noise=0.4` draws errors between −0.2 and +0.2 radians. The focal agent participates in the circular mean. All headings are calculated from one snapshot; positions advance using the old heading (the original backward update convention). No turn clipping, attraction or repulsion. |
| `boids` | **Boids - separation, alignment and cohesion - 2D.** One simple schooling model with a shared perception radius, minimum/maximum speed and the three behaviour weights. Separation strengthens at short distances; alignment matches neighbour velocities; cohesion pulls towards the local centre. Fixed gains and configurable speed bounds are documented below. No steering-force slider or separate simulator variants. |
| `voronoi` | **Voronoi-filtered zonal - 2D.** The existing CHIMES zone rules applied only to Delaunay/Voronoi neighbours, retaining metric cutoffs. This is not the Gautrais/Calovi turning-dynamics model. |

## Restricted options

New named model settings default to `restricted=True`. In the interactive GUI,
**Restrict to model options** hides the social-speed extension, advanced speed
controls, heading-noise memory and pursuit/fleeing/capture relationships. These
extensions are also suppressed by the movement engine, including when their
previous values remain in a loaded settings file. Disable restriction to use
compatible extensions; the selected model's defining equations remain in effect.
For example, Boids always uses steering acceleration and Vicsek always uses
uniform noise, rather than acquiring the zonal speed/noise-memory rules.

Restricted Couzin models and the overlapping-zone variant use omnidirectional repulsion and their own interaction
radii without the extra global visual-range cutoff. Restricted Boids likewise uses its
own perception radius and full angular perception for all three steering behaviours.
Disabling restriction enables the CHIMES field-of-view and detection-range extensions.
The visual-network overlay is
still a separate perception diagnostic and does not determine these social rules.
Arena boundaries and internal walls remain available as CHIMES environmental
extensions; they were not part of the original open-space Couzin scenarios.

Restricted Vicsek uses full angular perception, a periodic arena without walls,
the global speed for all agents, and its own interaction radius. The GUI switches
to periodic boundaries when selecting it. Existing walls are retained by selecting
extension mode instead; adding walls or bounding the arena also leaves restricted
Vicsek mode. Scripted restricted settings reject walls/bounded arenas explicitly.
A new `settings(source="defaults", model="vicsek")` scenario defaults to periodic.

Legacy `model="zonal"` files and agent dictionaries without a model retain the
previous extension behaviour (`restricted=False`), so existing scenarios still
load. They appear under the explicit Couzin 2002 2D label with restriction off.
For the current restricted defaults, use `source="defaults"` rather than your last
saved session. Old scientific and display settings are not overwritten by a code
upgrade.

## Scripted settings

```python
import chimesim

scenario = chimesim.settings(source="defaults", model="zonal_overlap",
    environment={"bounded": False},
    agents={"A1": {"n": 100, "r_r": 10, "r_o": 60, "r_a": 120,
                    "attraction_start": 25, "restricted": True}})
chimesim.interactive(settings=scenario)
```

Individual variation continues to work for the exposed parameters. An overlapping
model still requires `r_r <= r_o <= r_a`; `attraction_start` lies between `r_r` and
`r_a`. In the shared-zone model, `r_o` is retained in saved data for compatibility
but does not affect movement. Vicsek and Boids only use `r_a`;
Boids retains legacy `r_r` and `r_o` values in saved files but ignores them. Its speed multiplier sets a ceiling, not a compulsory speed. Its parameter names
are `separation_weight`, `alignment_weight`, `cohesion_weight`. The legacy
`max_acceleration` value is ignored.

## Implementation references and limits

- [Couzin et al. (2002), Collective memory and spatial sorting](https://doi.org/10.1006/jtbi.2002.3065).
  The original model was three dimensional. CHIMES retains its planar direction
  sums, Gaussian heading perturbation, repulsion priority and turn clipping.
- [Couzin et al. (2005), Effective leadership and decision-making](https://doi.org/10.1038/nature03236).
  The current option implements the uninformed social response, not the full
  leadership experiment. Informed and uninformed populations remain planned.
- [Vicsek et al. (1995), Novel type of phase transition](https://doi.org/10.1103/PhysRevLett.75.1226).
  The periodic geometry, inclusion of self, uniform angular noise, fixed speed and
  snapshot/position update convention are explicit. Interaction radius and noise
  may be varied among types as an extension to homogeneous reference scenarios.
- [Reynolds (1987), Flocks, herds, and schools](https://www.red3d.com/cwr/papers/1987/SIGGRAPH87.pdf)
  provides the flocking background, rather than a claim to reproduce 3D flight.
- [GeoPatternLab's published flocking implementation](https://geopatternlab.com/assets/Flocking-BjiUDGEy.js)
  supplies a checked example of the simple distance-based three-rule equations
  below. This is an implementation reference, not a separate selectable model
  or a claim to reproduce Reynolds' complete 3D flight mechanics.

## Boids: one simple schooling model

The controls are **Perception radius**, **Minimum speed**, **Maximum speed**,
**Separation**, **Alignment** and **Cohesion**. There are no named online-simulator variants.
For perceived neighbours at strictly positive distance below `r_a`, acceleration
is the sum of three documented terms:

- `0.008 * cohesion_weight * mean(neighbour displacement)`;
- `0.05 * alignment_weight * (mean(neighbour velocity) - own velocity)`;
- `12 * separation_weight * sum(away displacement / squared distance)`.

Separation retains its distance-dependent magnitude, so repulsion grows as
neighbours get closer instead of becoming a constant-strength steering request.
The terms are combined synchronously and added to existing velocity. Speed
is clipped to the configured minimum and maximum; minimum speed defaults to
zero, as stated in Reynolds' 1987 paper, page 3, Geometric Flight. The minimum
is independent of the maximum, not a fixed 45% ratio. With a positive minimum,
a stationary agent starts along its retained heading. Coincident neighbours
are excluded because their separation direction is undefined.

The gains are documented numerical choices from the checked example, not
newly fitted rules. This remains a simplified planar three-rule model: the
1987 paper also describes damping, banking and prioritised allocation of a
maximum acceleration. Those full geometric-flight mechanics are not reproduced
here. Removing a force-limit slider is a simplification, not evidence that
acceleration limits are absent from the original paper. Older `max_acceleration`,
`r_r` and `r_o` values remain loadable but do not affect this Boids implementation.

`movement.min_speed` and `movement.desired_speed` set the global speed interval
in model pixels per step. Both bounds use the same per-type/individual
`speed_scale` multiplier. Invalid bounds (negative/nonfinite minimum or minimum
above maximum) are rejected by the interface and scripted configuration.

Restricted mode uses full angular perception, its own radius and a periodic,
wall-free arena. Boundary, wall, capture and redundant visual-range controls
are hidden. Extension mode exposes the CHIMES field-of-view, detection cutoff,
wall and relationship features explicitly. Body size remains a separate
visual/perception setting above Colour by, not a personal-space force.

Load `examples/boids_schooling.json` using Load settings for a starting example:
160 agents, radius 55, minimum speed 2.025, maximum speed 4.5, separation 0.35, alignment 1 and
cohesion 2, in a 1000 × 1000 periodic arena. Its positive minimum is explicit in the preset and is
not the model default or tied automatically to maximum speed. It is a starting
configuration, not a promise of one particular flock shape. Density, initial headings and
interaction weights determine which schools form.

Initialisation retains CHIMES spawning and random-number generation. Cursor interaction defaults to Off. Selecting Repel or Attract opts into an
explicit external intervention, separate from the internal Boids equations. The comparison website's screenshot has Attract selected, which affects
motion while its pointer is over the canvas. Matching those six parameters
therefore does not imply matching that entire animation or its `FLOCK` seed.

Numerical fixtures execute the reference's original step closure from identical
initial states. Its source URL and SHA-256 are recorded in
`tests/fixtures/boids_reference.json`. The companion generator recreates the
fixtures from a local copy of the source module. Tests cover multi-step flocking,
periodic seams and exact overlaps, with the reference speed floor requested
explicitly for comparison. Separate tests verify the zero default and
independent configurable bounds. Independent schooling checks
verify that the model maintains travel and spacing instead of collapsing.

Per-type speed is displayed as **Maximum speed** in model pixels per step,
with the internal multiplier converted for display and editing. When agents
vary, sliders show means, explained once in the variation editor. With multiple
types, the perception control in each type window is a modifier relative to the
general radius; with one type it appears only in General movement settings.
