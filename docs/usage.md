# CHIMESim user guide

[Back to the overview](../README.md) · [Roadmap](../CHANGELOG.md#roadmap)

## Interactive exploration

```python
from chimesim import run

run(width=1200, height=800, window_width=900, window_height=600)
```

The GUI restores its last session from `~/.chimesim/settings.json`, falling back to previous ABSim settings on first launch. Controls start at the top left, General movement settings below them, and the simulation to their right. Agent-type windows cascade on the right; Information opens near the bottom of the simulation. Window positions, sizes and hidden panels are saved after moves settle and on exit. Closing Controls or Simulation exits the session. Close another settings panel to hide it; use the simulation window’s **Windows** menu to restore it. Closing an agent-type panel removes that type; closing the final type only hides its panel. GUI types are labelled A1, A2, etc.; use the **A1**, **A2** and **A3** checkboxes to include types (at least one remains active).

The Controls window contains Start/Pause/Reset, trajectory and settings export, arena size, the universal interaction-model selector and population sliders. The General movement settings window contains global movement and wall parameters; each agent window contains its own behavioural parameters. All GUI agent types use the selected interaction model. Information shows step count and FPS, current group scores and rolling plots for polarisation, cohesion and mean speed. Cohesion uses within-type nearest-neighbour distance (lower means more cohesive); distances respect periodic boundaries. Speed is the mean agent displacement per step. Each type retains its colour across the plots. A fourth plot shows spatial groups across all types (see below). Plot history contains the latest 180 display samples, sampled at most about seven times per second while steps advance; Reset clears it. Colour legends remain available for heterogeneous types. Scrollbars appear only when content needs them. Wall controls are disabled in periodic arenas without mask walls; wall-following is enabled for both outer walls and mask/drawn walls.

The geometric-neighbour overlay shows Delaunay geometry, not the complete directed interaction network after zone, visibility and relationship filters.

Use explicit agent types to start a custom scenario:

```python
from chimesim import BOID, run

slow = BOID.clone("slow")
fast = BOID.clone("fast")
slow.speed_scale = 0.8
fast.speed_scale = 1.3
run(agent_types=[slow, fast], bounded=True, seed=42)
```

Types without hunting/fleeing relationships school together. Changing the steps-per-frame control advances more model steps between drawings. `run(log=True)` returns metrics for each model step after the window closes; resetting clears that session log. Moving or resizing windows temporarily suspends simulation updates; a running simulation resumes when window movement settles, while a paused simulation stays paused. By default, resizing the simulation window proportionally changes arena dimensions and agent positions after dragging settles, rebuilds wall geometry, and updates the Width/Height fields. Select **Lock arena size when resizing** to keep the simulated space fixed and only scale the view. Use the lock when comparing experiments. The arena boundary is redrawn when the view changes, including while paused.

### Speed controls and spatial groups

**Speed multiplier** sets preferred speed as global speed × multiplier. Individual distributions can give agents different preferred speeds and social responses. The speed controls now separate three effects:

- **Social speed** (0–1): strength of the longitudinal social response. All perceived neighbours within the selected model's interaction range contribute, including the attraction zone. Far neighbours ahead accelerate an agent; far neighbours behind brake it. Close neighbours reverse that response to maintain spacing. Velocity differences also contribute. Directional projection makes side-by-side distance forces weak in speed and stronger in turning through the existing direction model.
- **Preferred spacing**: distance where the attraction/repulsion speed response changes sign. This is independent of the three direction-zone radii.
- **Speed recovery**: rate of return towards preferred speed, per model step. Larger values restore preferred speed faster; zero removes that restoring force while variable speed is enabled.
- **Speed noise**: independent temporal fluctuation amplitude, expressed as a fraction of preferred speed per square-root step. It defaults to zero. It can vary speeds even when Social speed is zero and there are no neighbours.
- **Max speed ×**: explicit speed ceiling as a multiple of preferred speed, default 3. Speeds cannot become negative.

With Social speed and Speed noise both zero, agents retain the original fixed-speed behaviour. A frozen speed distribution across agents (`set_distribution("speed_scale", ...)`) is distinct from temporal speed noise. The new parameters also support `set_distribution`, including heterogeneous `social_speed` strengths. Per-agent values sampled at birth override type sliders.

Social speed respects the same field of view, wall visibility and compatible-type rules as the direction model; Voronoi mode also restricts it to geometric neighbours. To respond to neighbours directly behind, set **Field of view** to **360°**. Increase **ZOA** (or **Max interaction distance**) for longer-range responses. The rolling mean-speed plot shows a group average: acceleration and braking can cancel in the mean while individual speeds vary strongly.

The speed update adapts the preferred-speed relaxation, temporal fluctuations and longitudinal projections of velocity-alignment and smooth distance-regulation forces in [Klamser et al. (2021), equations 7–11](https://doi.org/10.3389/fphy.2021.715996). It uses `tanh((distance − preferred_spacing) / preferred_spacing)` for distance regulation, scales its force by preferred speed and Social speed, and averages over perceived neighbours. Relaxation and noise are integrated exactly over one simulation step with the social force frozen at the shared starting snapshot. Speed is clipped at zero and the explicit speed ceiling.

This is an adaptation to CHIMESim's existing direction models, not a reproduction of the published model. In particular, the paper's coupled speed-dependent turning, continuous overlapping forces, open boundaries and published parameter/integration choices are not reproduced. Reproducing its figures requires a separate model implementation and validation.

The exploratory **Spatial groups** plot links agents closer than **Group distance × NND** times the median positive nearest-neighbour distance. This slider is in **Display**, defaults to **2.5**, updates the count immediately and is saved as `group_distance_multiplier`. Larger multipliers merge more components. Grouping uses all types together; connected components of at least two agents are groups, while singletons are isolated agents. It updates at most twice per second during running, and wraps distances across edges/corners in periodic arenas. It can merge groups through chains, is sensitive to different local densities and respects interior wall visibility; this remains an exploratory classifier. The current link distance is displayed alongside the group count.

### Individual positions and spatial leadership

In **Display**, enable **Show individual positions**. The **Individual positions** window has one horizontal colour strip per agent over time. Choose **Spatial leadership** (rear → front) or **Centre distance** (centroid → outermost member), and order rows by an assigned trait such as Speed multiplier or attraction range. Low trait values are at the top. Rows keep their agent IDs and ordering values; **Reorder rows** uses the latest trait values without recolouring or resampling agents. The Agents filter changes which types are displayed, not which members define a spatial group.

The view retains the latest 180 display samples, collected at most 2.5 times per second while this view or positional colouring is active. Columns represent successive samples; displayed steps and CSV step values retain their simulation-step timing. Reset and changing the grouping threshold clear history. **Export history** saves the retained samples, raw and relative position scores, sample-local spatial component labels and actual individual traits. It is separate from full trajectory recording.

**Spatial leadership** is signed displacement from the physical group's centroid along its mean unit heading: positive means in front. Its display score divides by the largest absolute projection in that group, giving −1 to +1. **Centre distance** is ordinary radial distance from that centroid, displayed relative to the most distant member (0–1). A member need not sit at the centroid, so the smallest radial score need not be zero. These quantify spatial position, not causal influence.

Groups use the same all-type spatial connectivity and multiplier as the group-count plot. Periodic groups are unwrapped across edges/corners before the centroid is calculated. Isolates have no within-group score. Front/back is unavailable for stationary or weakly aligned groups (polarisation below 0.2); radial distance remains available. Groups spanning half or more of a periodic dimension, or with an ambiguous circular centre, also receive unavailable positions. Grey marks these cases instead of inventing a leader or centroid.

The separate **Colour agents by** menu in Display offers **Traits / types**, **Spatial leadership** and **Centre distance**. These overlays leave individual traits and their Colour by selections intact. Calculations live in `metrics.py`; the optional history renderer is in `ui_positions.py`. `metrics.within_group_positions(agents, environment, multiplier=2.5)` exposes the same per-agent values for analysis, and `metrics.compute()` now includes mean speed for each type. Nearest-neighbour metrics use periodic-aware spatial trees rather than allocating a dense distance matrix.

### Arena dimensions versus window dimensions

| Parameter | Meaning |
|---|---|
| `width`, `height` | Actual simulation arena dimensions in model pixels; defaults are 1920 × 1080 without saved settings |
| `window_width`, `window_height` | Initial visualisation window dimensions in screen pixels; defaults fit the arena view to the screen |

All four values can be supplied to `run()` or in its settings dictionary/JSON. Explicit arguments override saved values. The arena follows simulation-window resizing unless **Lock arena size when resizing** is selected. Other windows never change the arena. To set exact dimensions, edit **Width** and **Height** in the Controls window and click **Apply size & reset**. This rebuilds walls at the new arena size, resets the population, metrics log and recorded trajectories, and leaves the simulation paused.

Saved GUI settings retain both arena and display sizes. `simulate()` uses the arena width/height and ignores the display dimensions, so a configuration keeps the same physical space in GUI and headless runs.

### Storing simulations

**Start recording** collects agent positions and headings from that moment, including every model step between drawings. **Stop recording** ends collection; starting again appends to the same recording. Recording is off when the GUI opens. **Export recorded data** saves everything collected so far to CSV without clearing it or starting recording. Reset and Exit discard the temporary recording, so export first. Data are buffered to a temporary file rather than accumulated in memory.

Export also writes `<trajectory-name>-traits.csv`, containing each agent’s actual trait assignments at recording start and subsequent changes/new agents. Rows identify agent ID, group, parameter, value, step, next effective step and sample; angles use radians. This complements distribution settings with the values actually used.

Export also writes `<trajectory-name>-parameters.csv`, with the initial configuration and subsequent parameter changes while recording. Each row includes the completed `step`, the next `applies_from_step`, recording `sample`, event, parameter path and old/new JSON values. Recording start/stop events identify gaps; resuming records the current configuration differences. Settings changed while recording is off are captured on resuming, not at their original change time. This is an audit of effective parameter changes, not an exact replay file.

CSV columns are `sample`, `step`, `id`, `group`, `x`, `y`, `heading_deg`, `dx`, `dy`, `width` and `height`. Headings use degrees; positions and velocities use model pixels and pixels per step. `sample` distinguishes additional snapshots at the same step. **Load settings** opens a JSON file picker and starts a fresh paused session with that configuration. Invalid files leave the current session intact; cancel the picker to continue. If recorded data exist, loading asks before clearing them so you can cancel and export first. Relative mask paths resolve beside the selected JSON file. **Save settings** writes a JSON configuration for `run()` or `simulate()`; it recreates parameters, not the exact agent state. Window sizes, positions and visibility save automatically after moves settle and on exit, independently of this button.

The **Display** section toggles neighbours, Information and trajectory trails. **Trail length** controls the visible history; trails are independent of data recording.

## Repeated headless simulations

```python
from chimesim import simulate

results = simulate(n_steps=1000, n_reps=5, burn_in=200, seed=42,
                   save_trajectories=True)
```

`n_steps` is the number of recorded steps **after** burn-in, for each repeat. Repeats use `seed + rep`. Headless defaults are independent of the GUI auto-save, and runs use their own random generator.

Results contain:

- `summary`: one dictionary per recorded step and repeat, with group counts, polarisation, nearest-neighbour distance and overall polarisation.
- `trajectories`: when requested, per-agent position, heading and sampled individual traits. Identify an agent using `(rep, id)`; IDs are not reused within a repeat.
- `settings`: the complete resolved configuration, which can be saved as JSON and reused.

Rows can be converted to tables if pandas is installed:

```python
import pandas as pd

summary = pd.DataFrame(results["summary"])
trajectories = pd.DataFrame(results["trajectories"])
```

Trajectories are stored in memory; recording every agent at every step can become large. A seed reproduces a run with the same package implementation, dependencies and configuration, rather than guaranteeing identical results across future numerical-library versions.

## Configurations as parameters, dictionaries or JSON

Explicit function arguments override settings; omitted arguments use settings, then package defaults. Settings are never implicitly merged with the last GUI session.

```python
experiment = {
    "width": 800,
    "height": 600,
    "bounded": True,
    "desired_speed": 3.0,
    "fov_deg": 270,
    "n_steps": 1000,
    "burn_in": 200,
    "n_reps": 5,
    "seed": 42,
    "agents": {
        "fish": {
            "n_default": 60,
            "r_r": 10,
            "r_o": 30,
            "r_a": 75,
            "noise": 0.1,
            "interaction_model": "zonal",
            "distributions": {
                "speed": {"dist": "uniform", "low": 0.8, "high": 1.2}
            }
        }
    }
}
results = simulate(settings=experiment, n_reps=10)
```

Save and reuse a complete configuration:

```python
import json
from pathlib import Path

Path("experiment.json").write_text(json.dumps(results["settings"], indent=2))
repeated = simulate(settings="experiment.json")
```

Relative mask paths in a JSON file are resolved relative to that file. Unknown settings and unreadable files raise errors. To load GUI settings deliberately:

```python
results = simulate(settings="~/.chimesim/settings.json", seed=42)
```

The loader also accepts GUI names `speed`, `fov`, `n`, `max_turn_deg` and `noise_deg`. `run(settings="experiment.json")` can open the agent and movement configuration in the GUI; the arena uses the saved width/height independently of the visualisation window size. Run length and repeats apply to headless runs.

## Individual variation and agent parameters

Each agent-type window now has an **Individual variation** editor. Choose a trait, then:

- **Fixed**: all agents use the main slider value.
- **Normal**: the main slider sets the bell curve's centre, **Spread (SD)** its width, and optional minimum/maximum bound it. Draws come from a bounded normal distribution instead of being pushed onto the endpoints. With asymmetric bounds, the resulting mean can differ from the bell curve's centre.
- **Uniform**: enter a minimum and maximum; applying sets the main slider to their midpoint. Moving that slider shifts the range; near physical limits the symmetric range narrows.

**Apply variation** updates the selected trait for existing agents, keeping their positions, headings and other traits. Moving a main slider subsequently updates that trait's central value and resamples it; affected sliders are marked **mean**. Grey preview bars show current agents and the line shows the proposed distribution. Multiple traits can vary independently. Zone radii must remain ordered per individual; bounds are restricted by adjacent radii, and invalid changes leave the previous configuration intact.

The independent **Colour by** selector can show any supported trait, or restore **Type colour**. It does not resample values. A varying trait creates a gradient; a fixed trait gives one uniform colour. Colours share a scale across current agents of the same type, including newly added agents. **Speed multiplier** colouring shows that assigned trait, not instantaneous movement speed.

Saved settings retain distribution definitions and colour selection; starting or resetting a simulation samples new individuals. Existing Python configurations with lognormal/bimodal distributions remain supported; choosing Fixed, Normal or Uniform in the GUI replaces the selected distribution.

```python
from chimesim import BOID, simulate

fish = BOID.clone("fish")
fish.set_distribution("speed", dist="uniform", low=0.8, high=1.2)
fish.set_distribution("zoa", dist="bimodal", means=[60, 100], stds=[5, 5],
                      clip=(35, 150))
results = simulate(agent_types=[fish], n_steps=500, seed=42)
```

Distributions support normal, uniform, lognormal and bimodal sampling. Supported individual parameters are `speed_scale`, `r_r`, `r_o`, `r_a`, `flee_dist`, `max_turn`, `noise`, `social_speed`, `preferred_distance`, `speed_relaxation`, `speed_noise` and `max_speed_scale`. Aliases include `speed`, `zor`, `zoo`, `zoa`, `detect`, `maxturn` and `socspeed`.

API distributions retain their existing clipping semantics by default. For a bounded normal without endpoint piles, use `set_distribution("speed", mean=1, std=.2, clip=(.5, 1.5), truncate=True)`. Physical limits still apply: nonnegative values, `social_speed <= 1`, `max_turn <= π`, positive preferred distance, and `max_speed_scale >= 1`. Sampled radii must satisfy `r_r <= r_o <= r_a`; use suitable bounds. Individual values stay fixed between explicit GUI changes or calls to `apply_variation`; they override type defaults during movement. Resetting samples new individuals.

| Parameter | Meaning |
|---|---|
| `n_default` | Initial agent count |
| `r_r`, `r_o`, `r_a` | Ordered repulsion, orientation and attraction outer radii |
| `speed_scale` | Multiplier on `desired_speed` |
| `max_turn` | Maximum social heading update per step, radians |
| `noise` | Gaussian heading-noise standard deviation, radians; applied before turn clipping |
| `social_speed` | Strength of social acceleration/braking, 0–1 |
| `preferred_distance` | Positive spacing where the distance-dependent speed response changes sign |
| `speed_relaxation` | Preferred-speed recovery rate per step, default 0.15 |
| `speed_noise` | Temporal speed noise / preferred speed, default 0 |
| `max_speed_scale` | Maximum speed / preferred speed, default 3, minimum 1 |
| `flee_dist` | Threat detection distance |
| `interaction_model` | `"zonal"` or `"voronoi"` |
| `hunts`, `flees` | Names of other types to hunt or flee |

Positions, radii and distances use pixels; speeds use pixels per model step. Python parameters use radians for `noise` and `max_turn`; the GUI displays degrees. `fov_deg` always uses degrees. Wall and predator steering are applied after the social turn limit.

## Interaction models and environments

The **zonal** model gives repulsion priority; otherwise agents combine neighbour headings in the orientation zone and attraction toward neighbours in the outer zone. The **Voronoi** model restricts those rules to geometric neighbours, with `r_a` as the maximum interaction distance. Small populations remain interactive; degenerate triangulations fall back to zonal rules. Periodic arenas use tiled triangulation. Agent movement is calculated from a shared starting state and applied together.

Bounded arenas have outer walls; periodic arenas wrap at their edges. Metrics and hunting/fleeing use the same boundary convention. Predation is evaluated before movement, so a capture between two recorded positions is not detected until the next step.

For image walls, install the `masks` extra and provide `mask="arena.png"`. Dark source pixels become solid grid cells. Walls use an approximately four-pixel grid, so thin walls are preserved but may be widened. Visibility checks traverse the complete segment through that grid, and movement is stopped before crossing a solid cell. A blocked move reverses the heading; the agent is treated as a point for collisions, regardless of its drawn size.

Smooth wall avoidance is controlled by `wall_avoid_radius` and `wall_strength`. `thigmotaxis` adds a tangential steering force near both outer rectangular walls and mask/drawn walls, aligned with the current direction of travel. It acts within the wall-avoidance radius and requires positive wall strength; it does not attract agents towards walls. At outer corners the nearest edge determines the tangent. Hard collision handling remains a fallback if an agent reaches a wall. Mask geometry is stretched to fit the arena.

For lower-level work, `Model`, `Environment`, `AgentType`, `Boid` and `metrics` are public. Attach an environment before initialising agents, and pass it to standalone metrics: `metrics.compute(model.groups, model.env)`. A model without an environment uses periodic boundaries. Direct nearest-neighbour metrics without an environment retain the default periodic convention.

## Tutorial and development

[CHIMESim_tutorial.ipynb](../CHIMESim_tutorial.ipynb) provides interactive examples and lower-level stepping. Start with the examples above for repeatable headless experiments.

Run the focused model checks with `python -m unittest discover -s tests`. These cover social acceleration/braking, temporal noise, heterogeneity, speed limits and periodic grouping.

Future development is listed in the [roadmap](../CHANGELOG.md#roadmap).

## Maintainer and licence

Developed and maintained by Jolle Jolles. Ruiheng Jin contributed implementation work during an internship. Earlier contributions remain recorded in the Git history.

Released under the [MIT licence](../LICENSE).

### Creating interior walls

In **Controls → Arena**, **Load mask** opens an image picker. Choose whether to resize the arena to the image dimensions or stretch the image to the current arena. Dark pixels are solid, opaque walls. **Remove mask** removes the image layer and preserves drawn walls.

Choose **Draw walls: Brush, Rectangle or Ellipse**, then drag in the simulation window. Rectangles and ellipses are filled obstacles; Width sets brush thickness in arena pixels. Drawing pauses the simulation and commits each gesture on release. Agents covered by new walls move to nearby free space. Select **Off** when finished, then Start to resume. **Undo wall** removes the last drawing; **Clear drawn walls** preserves the image layer. Drawings scale with arena size and are stored in settings as normalised `wall_shapes`; saved settings also work with `simulate()`. Image masks remain external files. Environment edits appear in the recording parameter log.

Spatial groups only link nearby agents when the direct, shortest periodic sight line is clear. This also affects within-group centroids and spatial leadership. Agents connected around the end of a wall can still form one group.

Zone-radius distributions are sampled conditional on repulsion ≤ orientation ≤ attraction for every individual. Overlapping distributions therefore have effective means and spreads that can differ from their requested values. Rejected individuals resample only the edited radius traits; no clipping or sorting is used. Configurations with no valid samples within 128 retries raise an error.

## Related software to review

- Compare scope, interfaces and model implementations in [ninjahawk/swarmsim](https://github.com/ninjahawk/swarmsim), [malintha/swarmsim](https://github.com/malintha/swarmsim), [diBernardoGroup/SwarmSimPublic](https://github.com/diBernardoGroup/SwarmSimPublic) and [yxiao1996/SwarmSim](https://github.com/yxiao1996/SwarmSim); identify useful ideas and differences before considering code reuse.

## Camera and agent shapes

Use the mouse wheel over the arena or **Display → View −/+** to zoom. Pan with the scrollbars or drag with the right/middle mouse button. **Fit arena** shows the whole arena again. Zooming and panning change only the view. Select **Lock arena size when resizing** explicitly if window resizing should also preserve the physical arena. Drawing continues to use arena coordinates at every zoom level.

**Agent shape** selects Triangle or Circle with tail and is saved in settings. Both use the existing body-size parameter in model coordinates. The circle is the body silhouette; its tail indicates heading. The independent perception API uses the same unscaled body geometry for agent occlusion. Current movement models still use point-to-point wall visibility; they do not yet use that perception API.

Choose **Draw walls → Fill** and click a region to toggle it between solid black and white free space. Brush walls and arena edges enclose regions; the fill tool treats edges as closed even in periodic arenas. Clicking a filled region again restores its previous footprint to white without erasing its enclosing strokes. Fills use the collision grid, so small gaps must be closed at that resolution. Undo removes the last fill or drawing. An edit that leaves no free cells is rejected. Agents start only in white cells; existing agents covered by edits are relocated to free space. Fills persist in saved settings.

## Visual perception (Python API)

Perception is opt-in and does not change movement or grouping. The GUI offers a visual-neighbour overlay. Take a snapshot from a lower-level `Model`:

```python
from chimesim import BOID, Model, Environment, perception

fish = BOID.clone("fish")
model = Model([fish], seed=42)
model.env = Environment(800, 600, bounded=True)
model.init_agents("fish", 30)
view = perception.perceive(model.all_agents, model.env,
                          body_style="Circle with tail", fov_deg=270,
                          max_distance=200)
```

All output matrices use observer rows and target columns, aligned with `view.ids`:

| Output | Meaning |
|---|---|
| `distance` | Centre-to-centre distance, minimum-image in periodic arenas |
| `angular_extent` | Full apparent body width in radians, before occlusion/FOV |
| `in_fov_angle` | Body width within the observer's field of view, before occlusion |
| `visible_angle` | Angular width remaining after FOV, walls and nearer bodies |
| `visible_fraction` | `visible_angle / angular_extent`, or zero for no angular extent |
| `visible` | Whether any sampled portion is visible |

Triangle is the default; `body_style` can also be a per-agent list. Body size comes from `agent.type.size`. Display magnification and tail length do not affect perception; tails are not opaque bodies. Each observer looks along its movement heading, retaining its stored heading while stationary. `fov_deg` can be scalar or per-observer. All types can occlude one another regardless of hunting/schooling relationships. No behavioural weights are inferred from visibility.

The calculation integrates target-centred angular rays, defaulting to at least 32 samples per in-FOV silhouette segment and a maximum angular spacing of 0.5 degrees. Increase `samples_per_body` or decrease `angular_resolution_deg` to resolve narrower occlusions. This is a numerical approximation, not an exact visibility polygon. Without occlusion, angular widths/FOV clipping are geometric; occlusion fractions converge with sampling resolution. Use `agent_occlusion=False` to inspect wall/FOV effects alone.

`max_distance` is an optional scalar or per-observer centre-distance cutoff for targets and body occluders; omitted means unlimited. Angular outputs outside it are zero. Use a cutoff for large snapshots; this API is not run automatically every simulation frame. Matrix outputs use O(N²) memory; temporary ray/body matrices are processed in batches. Walls retain the existing computational grid. Without an environment, coordinates are ordinary nonperiodic space. With a periodic environment, only the nearest image of each body is considered, appropriate for local perception rather than arena-spanning bodies/views. Equal-depth overlapping bodies do not occlude each other; overlapping eyes/bodies are degenerate scenes, not a contact model.

### Visual-neighbour overlay

In **Display**, select **Show visual neighbours (directed)**. Arrows run from observer to observed neighbour; darker links indicate a greater visible fraction. Faint grey stippled links run centroid-to-centroid, above geometric links and below agents; arrows indicate direction. **Visual range** in General movement settings is the base centre-distance cutoff for perception. Each agent-type window has **Detection range ×**, multiplying that base; zero disables that type’s visual detection. **Threat detect range** remains the independent fleeing-distance parameter used by the current movement models. **Minimum visible fraction** filters displayed links. The overlay uses the selected triangle/circle body style, physical body size and global field of view, including wall and agent occlusion. It does not apply social-model or hunting/schooling filters. Display magnification and tails do not affect visibility.

Perception runs in a background snapshot calculation, at most twice per second; link membership can lag moving agents. Pausing lets the pending snapshot complete. Geometric and visual overlays can be toggled independently. Visual links do not yet change movement or spatial grouping.

### Display scale and frame rate

**Show scale bar and FPS** toggles both overlays: FPS at the top left and a single unlabelled 100-unit bar at the bottom left, with ticks at 0, 25, 50, 75 and 100. The bar follows zoom. **Triangle size / tail length ×** is visual only: triangle length matches circle diameter plus tail at the same setting, while the circle radius stays fixed.
