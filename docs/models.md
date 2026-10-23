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
| `boids` | **Boids - Reynolds-style steering - 2D.** Separation radius, neighbourhood radius, separation/alignment/cohesion strengths, maximum acceleration and speed ceiling. Steering acceleration changes the existing velocity, rather than setting a desired fixed-speed heading. Requests consume the acceleration budget in order: separation, velocity matching, centring. No artificial heading noise or zonal social-speed extension. |
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
own neighbourhood radius and one common FOV for all three steering behaviours.
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
but does not affect movement. Vicsek only uses `r_a`. Boids uses `r_r` and `r_a`;
its speed multiplier sets a ceiling, not a compulsory speed. Its parameter names
are `separation_weight`, `alignment_weight`, `cohesion_weight`, `max_acceleration`.

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
- [Reynolds (1987), Flocks, herds, and schools](https://www.red3d.com/cwr/papers/1987/SIGGRAPH87.pdf).
  This is a Reynolds-inspired planar steering implementation, not his complete 3D
  geometric flight model. Separation sums inverse-square-distance vectors;
  alignment requests the mean neighbour velocity minus current velocity; centring
  requests motion towards the local centroid at the speed ceiling. Weighted
  requests are allocated sequentially within the acceleration budget. Movement
  then applies CHIMES wall handling and a speed ceiling; no flight banking or
  aerodynamic dynamics are implied.
