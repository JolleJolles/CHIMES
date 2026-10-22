# CHIMESim user guide

[Back to the overview](../README.md) · [Roadmap](../CHANGELOG.md#roadmap)

## Interactive exploration

```python
from chimesim import interactive

interactive(width=1200, height=800, window_width=900, window_height=600)
```

The GUI restores its last scientific scenario from `~/.chimesim/scenario.json` and its window/display preferences from `~/.chimesim/display.json`. Older combined CHIMESim/ABSim settings migrate automatically. Controls start at the top left, General movement settings below them, and the simulation to their right. Agent-type windows cascade on the right; Information opens near the bottom of the simulation. Window positions, sizes and hidden panels are saved after moves settle and on exit. Closing Controls or Simulation exits the session. Close another settings panel to hide it; use the simulation window’s **Windows** menu to restore it. Closing an agent-type panel removes that type; closing the final type only hides its panel. GUI types are labelled A1, A2, etc.; use the **A1**, **A2** and **A3** checkboxes to include types (at least one remains active).

The Controls window contains Start/Pause/Reset, trajectory and settings export, arena size, the universal interaction-model selector and population sliders. The General movement settings window contains global movement and wall parameters; each agent window contains its own behavioural parameters. All GUI agent types use the selected interaction model. Information shows step count and FPS, current group scores and rolling plots for polarisation, cohesion, mean speed and milling. Cohesion uses within-type nearest-neighbour distance (lower means more cohesive); distances respect periodic boundaries. Speed is the mean agent displacement per step. Each type retains its colour across the plots. A fifth plot shows spatial groups across all types (see below). Milling is sampled alongside spatial grouping at most twice per second: each type’s rotational order is measured around each physical group’s centroid, then magnitudes are averaged with member-count weights. Opposite rotations in separate groups therefore do not cancel. Isolates and ambiguous periodic centroids are excluded, with gaps when no valid grouped agents remain. M near 1 indicates consistent rotation within groups; M near 0 indicates little net rotation. The same values appear as M beside each type’s scores. Plot history contains the latest 180 display samples, sampled at most about seven times per second while steps advance; Reset clears it. Colour legends remain available for heterogeneous types. Scrollbars appear only when content needs them. Wall controls are disabled in periodic arenas without mask walls; wall-following is enabled for both outer walls and mask/drawn walls.

The geometric-neighbour overlay shows Delaunay geometry, not the complete directed interaction network after zone, visibility and relationship filters.

Use a shared scenario to start a custom population:

```python
import chimesim

scenario = chimesim.settings(source="defaults", agents={
    "A1": {"n": 80, "speed_scale": 0.8, "color": "#56b1fc"},
    "A2": {"n": 40, "speed_scale": 1.3, "color": "#f4a030"},
})
chimesim.interactive(settings=scenario, seed=42)
```

Types without pursuit, fleeing or capture relationships school together. Changing the steps-per-frame control advances more model steps between drawings. `interactive(log=True)` returns metrics for each model step after the window closes; resetting clears that session log. Moving or resizing windows temporarily suspends simulation updates; a running simulation resumes when window movement settles, while a paused simulation stays paused. By default, resizing the simulation window proportionally changes arena dimensions and agent positions after dragging settles, rebuilds wall geometry, and updates the Width/Height fields. Select **Lock arena size when resizing** to keep the simulated space fixed and only scale the view. Use the lock when comparing experiments. The arena boundary is redrawn when the view changes, including while paused.

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

All four values can be supplied to `interactive()` or in its settings dictionary/JSON. Explicit arguments override saved values. The arena follows simulation-window resizing unless **Lock arena size when resizing** is selected. Other windows never change the arena. To set exact dimensions, edit **Width** and **Height** in the Controls window and click **Apply size & reset**. This rebuilds walls at the new arena size, resets the population, metrics log and recorded trajectories, and leaves the simulation paused.

Saved GUI settings retain both arena and display sizes. `simulate()` uses the arena width/height and ignores the display dimensions, so a configuration keeps the same physical space in GUI and headless runs.

### Storing simulations

**Start recording** collects agent positions and headings from that moment, including every model step between drawings. **Stop recording** ends collection; starting again appends to the same recording. Recording is off when the GUI opens. **Export recorded data** saves everything collected so far to CSV without clearing it or starting recording. Reset and Exit discard the temporary recording, so export first. Data are buffered to a temporary file rather than accumulated in memory.

Export also writes `<trajectory-name>-traits.csv`, containing each agent’s actual trait assignments at recording start and subsequent changes/new agents. Rows identify agent ID, group, parameter, value, step, next effective step and sample; angles use radians. This complements distribution settings with the values actually used.

Export also writes `<trajectory-name>-parameters.csv`, with the initial configuration and subsequent parameter changes while recording. Each row includes the completed `step`, the next `applies_from_step`, recording `sample`, event, parameter path and old/new JSON values. Recording start/stop events identify gaps; resuming records the current configuration differences. Settings changed while recording is off are captured on resuming, not at their original change time. This is an audit of effective parameter changes, not an exact replay file.

CSV columns are `sample`, `step`, `id`, `group`, `x`, `y`, `heading_deg`, `dx`, `dy`, `width` and `height`. Headings use degrees; positions and velocities use model pixels and pixels per step. `sample` distinguishes additional snapshots at the same step. **Load settings** opens a JSON file picker and starts a fresh paused session with that configuration. Invalid files leave the current session intact; cancel the picker to continue. If recorded data exist, loading asks before clearing them so you can cancel and export first. Relative mask paths resolve beside the selected JSON file. **Save settings** writes a JSON configuration for `interactive()` or `simulate()`; it recreates parameters, not the exact agent state. Window sizes, positions and visibility save automatically after moves settle and on exit, independently of this button.

The **Display** section toggles neighbours, Information and trajectory trails. **Trail length** controls the visible history; trails are independent of data recording.

## Scenarios and dataset generation

The supported top-level interface is `settings()`, `interactive()` and `simulate()`. A scenario is an ordinary JSON-compatible dictionary containing `model`, `environment`, `movement` and `agents`. No construction of `Model`, `Boid` or `Environment` objects is required.

```python
import chimesim

scenario = chimesim.settings(
    source="defaults", model="zonal",
    environment={"width": 1200, "height": 800, "bounded": True},
    movement={"desired_speed": 3, "fov_deg": 270, "visual_range": 200},
    agents={"A1": {"n": 100, "repulsion": 10, "orientation": 30,
                   "attraction": 75, "speed_scale": 1}},
)
chimesim.interactive(settings=scenario)
results = chimesim.simulate(settings=scenario, n_steps=1000, burn_in=200,
                            n_reps=5, seed=42, save_trajectories=True)
```

`settings(source="defaults")` starts from package defaults. `settings()` restores the last interactive scenario, falling back to defaults. Pass a dictionary or JSON path as `source` to edit that configuration; supplied environment/movement fields override the source, and supplied agents replace the population. Returned dictionaries are independent copies, so modifying one does not modify the source. `model` accepts `"zonal"` or `"voronoi"`. Population counts accept `n` or `n_default`; zone radii accept the descriptive names above or `r_r`, `r_o`, `r_a`. Angles use radians in agent definitions, with `noise_deg` and `max_turn_deg` aliases available.

With no explicit scenario, `simulate()` and `simulate(gui=True)` reuse the last interactive scenario. Explicit settings are combined only with package defaults, never with an unrelated previous session. For existing scripts, explicit scenario keyword overrides such as `width=800` also start from package defaults. Explicit run controls override any controls saved in an experiment file. Last-session window geometry and diagnostic display settings are not simulation inputs.

`burn_in=200` is the default; set `burn_in=0` to record immediately. `n_steps` counts recorded steps **after** burn-in, per repeat. `n_reps` sets repeats per condition. If no seed is supplied, one is generated and returned; repeat `rep` uses `seed + rep`. Runs use their own random generator.

Results contain:

- `summary`: one dictionary per recorded step and repeat, containing group metrics.
- `trajectories`: when requested, tracking-style dynamic states; one row per surviving agent per recorded frame.
- `individuals`: one row per initialised agent, with its type and effective assigned traits; includes agents lost during burn-in.
- `repeats`: one row per attempted repeat, with seed, recorded-step count, cancellation flag and time-averaged metrics. A repeat cancelled before recording has missing means. Online means exclude burn-in and do not require trajectories in memory.
- `settings`: the fully resolved scientific scenario.
- `run`: run controls including the actual seed.
- `conditions`: for a parameter grid, each condition's values, resolved scenario and starting seed; conditions and repeats are numbered from zero in data.

```python
import json
from pathlib import Path

configuration = {"scenario": results["settings"], "run": results["run"]}
Path("experiment.json").write_text(json.dumps(configuration, indent=2))
repeated = chimesim.simulate(settings="experiment.json")
```

Save a scenario alone to reuse its scientific settings with different run controls. Save the envelope shown above to reproduce both. Relative mask paths resolve beside the JSON file; old flat GUI/experiment files remain readable. Unknown fields raise errors. Reproducibility requires the same package implementation and numerical dependencies.

### Conditions and streaming output

```python
results = chimesim.simulate(
    settings=scenario,
    conditions={"agents.A1.speed_scale": [0.5, 1, 1.5], "agents.A1.n": [50, 100]},
    n_steps=1000, n_reps=5, burn_in=200, seed=42,
)
```

All combinations are run: here six conditions and five repeats each. Conditions use dotted scenario paths, including `movement.visual_range`, `environment.bounded`, `model`, agent traits and distribution fields. All combinations are validated before any simulation starts. Condition `c`, repeat `r` uses `seed + c*n_reps + r`. Distribution parameters take precedence over type defaults; vary `agents.A1.distributions.speed_scale.low`, for example, to change a configured distribution rather than its fixed default.

For larger datasets, use `output_dir="results"`. This streams CSV files into a new uniquely named run folder rather than retaining all rows in memory, and returns the folder and completion status. The folder contains configuration/status files and datasets described below; grids also include `conditions.json`. The experiment settings file can be passed back to `simulate(settings=...)` to repeat the same grid. Cancelled/failed runs preserve already written rows and record their status. Direct mode also supports progress and cancellation callbacks.

Use `chimesim.simulate(gui=True, settings=scenario)` or the installed `chimesim-simulate` command for the separate setup/progress GUI. Configure steps, repeats, output, population and arena, or load a saved experiment. The **Conditions** tab offers labelled parameter dropdowns (including loaded distribution fields) and comma-separated values or JSON lists; **Check combinations** reports the grid size. Three speeds and three attraction radii create nine conditions, with repeats for every condition. **Simulate** starts a background worker; **Stop simulation** stops between model steps and preserves written data. **Exit** closes the window and offers to stop an active run. After exit, `simulate(gui=True)` returns the most recent run information, or `None` if nothing ran. Closing this window does not open or control the interactive visualiser.

To plot a GUI run after exiting, use `saved = chimesim.simulate(gui=True)`, then `plot(saved, "polarisation", category="rep", time_bin=25)` after importing `plot` from `chimesim.plotting`. Existing run folders work directly too: `plot("/path/to/run-folder", "polarisation", category="rep")`; `analysis.load("/path/to/run-folder")` exposes all saved tables without rerunning the simulation.

Rows returned to Python can be converted with `pandas.DataFrame(results["summary"])`. In-memory trajectories can become large; prefer streaming output for long experiments.

### Output folders and data formats

`output_dir` is a **parent directory**, not a CSV filename. Relative paths are relative to Python's current working directory (`Path.cwd()`), which may differ from the notebook's location. Absolute paths make the destination explicit. Every call creates its own `run-<timestamp>` subfolder without overwriting earlier runs; all conditions and repeats of that call share the folder.

```python
from pathlib import Path

run = chimesim.simulate(
    settings=scenario, conditions={"agents.A1.n": [50, 100, 200, 400]},
    burn_in=200, n_steps=1000, n_reps=3, seed=42,
    save_trajectories=True, output_dir=Path.home() / "CHIMESim-results",
)
print(run["output_dir"])  # Absolute path to the new run folder
```

The older `output` argument remains an alias for `output_dir`. Omitting both returns Python datasets without writing files. For large runs and R workflows, prefer streaming to disk. Returned metadata contains the run-folder path and status; it does not retain copies of streamed datasets.

| File | Rows and contents |
|---|---|
| `trajectories.csv` | Optional; dynamic state per individual per recorded frame |
| `individuals.csv` | Initialised individual IDs, types and actual effective traits, sampled or fixed |
| `summary.csv` | Group metrics at each recorded step |
| `repeats.csv` | Arithmetic time means of those metrics for each repeat, with seed and completion information |
| `settings.json` | Resolved scenario, run controls, seed and requested condition grid |
| `conditions.json` | For grids, resolved scenario and starting seed per condition |
| `data_format.json` | Format version, identity keys, units and time conventions |
| `status.json` | Completion/cancellation/failure and completed workload |
| `performance.json` | Optional timings, throughput and computer details |

CSV files appear when their first rows are available. A repeat cancelled during burn-in can still have individual assignments and a repeat row, but no trajectories or per-frame summaries. Static traits are written once, rather than repeated millions of times. Join trajectories to individuals using **`set` and `id`**, or equivalently condition/repeat/id. `set` is `c<condition>_r<repeat>` and is unique within one run folder; prefix it with the folder/run identifier when combining separate runs.

Trajectory columns are:

| Columns | Meaning |
|---|---|
| `set`, `condition`, `rep`, `id` | Independent recording, zero-based condition/repeat and individual identity |
| `frame`, `t` | Completed model-step count, including burn-in; first exported frame is `burn_in + 1` |
| `x`, `y` | Model coordinates; y increases downwards when rendered |
| `vx`, `vy`, `speed` | Model velocity components and magnitude, in model units per step |
| `heading_rad` | Agent heading in radians, modulo 2π; zero points along +x, increasing towards +y |

Here `t` is numeric simulation time in model steps, not measured seconds. No calendar timestamps or physical calibration are invented. In periodic arenas, positions are wrapped. Recorded model velocity includes post-collision reflection and can differ from displacement-derived velocity at a wall. External analyses must account for periodic boundaries when calculating positional differences or distances; a generic tracking analysis may otherwise infer false jumps at arena edges.

The `set`, `t`, `id`, `x`, `y` structure follows the raw-input structure described in [swaRmverse's data-preparation vignette](https://marinapapa.github.io/swaRmverse/articles/step1_data_preparation.html). This is a plain CSV, not an R `trackdf` object; use its documented formatting step and specify the sampling period/context. Keep each simulation repeat separate when converting to tracks. Direct R import and trait joining need no package-specific loader:

```r
run_folder <- "/absolute/path/to/CHIMESim-results/run-..."
tracks <- read.csv(file.path(run_folder, "trajectories.csv"))
individuals <- read.csv(file.path(run_folder, "individuals.csv"))
repeat_summary <- read.csv(file.path(run_folder, "repeats.csv"))
tracks_with_traits <- merge(tracks, individuals, by = c("set", "id", "condition", "rep"),
                            all.x = TRUE, sort = FALSE)
```

CHIMESim provides simulation outputs and a small set of online diagnostic metrics. Subsequent scientific analysis and plots can stay in R, Python or another package. Use `save_trajectories=False` for summaries without raw tracks; `repeats.csv` is still calculated during the run. Keeping trajectories supports later analyses that were not anticipated when the simulation was configured. No full analysis pipeline is required inside the simulator.

### Timing and performance comparisons

Add `performance=True` to `simulate()` for a printed summary and `results["performance"]`. With `verbose=False` the report is still returned. Streaming output also saves `performance.json`; the setup GUI offers **Save performance report**.

```python
results = chimesim.simulate(
    settings=scenario, conditions={"agents.A1.n": [50, 100, 200, 400]},
    n_steps=1000, burn_in=200, n_reps=3, seed=42, performance=True,
)
for condition in results["performance"]["conditions"]:
    print(condition["initial_agents"], condition["steps_per_minute"])
```

Each condition reports elapsed wall time, process CPU time, completed steps, initial population, agent updates and throughput, with repeat-level timings and seeds. One step advances the whole population; agent updates count live agents before each completed step. Burn-in counts towards both workloads. Cancellation includes only work completed, including partial repeats.

Rates include population/environment setup, burn-in, group metrics, data collection/writing and callbacks. They exclude GUI setup, configuration validation and final report/file finalisation. Keep recording/output options the same across comparisons. Small runs are strongly affected by setup costs; longer runs and repeat-level rates help assess variability. CPU time covers this process and its threads; wall time measures how long the user waits.

The report includes computer name, operating system, processor/architecture, logical CPU count, total physical RAM when available, Python and package versions. Unavailable hardware fields are `None`. RAM describes installed memory, not simulation memory usage, and CPU count does not imply that the simulator uses all cores. No extra reporting dependency is required.

## Starter processing and plots

Optional dependencies: `python -m pip install -e '.[analysis]'`. The existing notebook extra also includes them. Core simulation and Tk windows do not import pandas or Matplotlib.

```python
from chimesim import analysis
from chimesim.plotting import plot

processed = analysis.process(results)  # Or analysis.process(run["output_dir"])
plot(processed, "polarisation", kind="time", category=["condition", "agent_type"])
plot(processed, "centroid_distance", kind="box", category=["condition", "agent_type"])
plot(processed, "speed", kind="density", category="condition")
plot(processed, "front_rank", kind="scatter", x="speed_scale", table="individuals")
```

`analysis.process` needs saved trajectories and individual assignments. It reconstructs physical groups across all types with the existing wall- and periodic-aware proximity rule. By default it selects the largest group in each frame; **ties include all equally large groups**, and isolated individuals are excluded. Set `groups="all"` to include every component of at least two members. `multiplier` and optional fixed `link_distance` control the grouping rule. Component numbers are frame-local, so they should not be interpreted as persistent group IDs.

The returned tables are:

- `groups`: per-frame polarisation, NND, mean centroid distance, mean speed, speed SD, actual centroid speed, milling, signed rotation and elongation. Type-specific rows describe each type within a selected physical group; mixed groups also have an `All` row. Centroid speed is measured from consecutive recorded centroids and requires unchanged membership. Its first value and values following membership changes are missing. Periodic centroids that cannot be unambiguously unwrapped are also missing.
- `positions`: each selected individual's front–back and centre rank, physical centroid distance and sample context. Scores are relative to the whole physical group, across types. Front rank is rear **0** to front **1**, using average ranks for ties. Centre rank is innermost **0** to outermost **1**. Front rank is missing for weakly polarised/stationary groups; centre distance/rank can still be available.
- `individuals`: each individual's mean ranks/distances over valid selected-group observations, observation counts, type and assigned traits. `front_rank` is the time-averaged spatial-leadership score, independent of whether a trait is heterogeneous. `front_samples` distinguishes unavailable front scores from actual observations.
- `structuring`: per-repeat SD among individuals' mean front ranks, plus the number of included individuals. This is descriptive, not a significance test; sparse or unequal coverage and temporal autocorrelation affect it.

For exchangeable identical agents that thoroughly mix over a long run, each individual's mean front rank is expected to approach **0.5**. Identical rules do not force equal realised positions, especially over finite runs. The mean of a complete frame's individual ranks is always 0.5, so that group average cannot measure structuring. Persistent differences among individuals' time averages are the useful result. No split-window consistency statistic or trait correlation is imposed.

**Mean centroid distance** is the arithmetic mean distance to the physical centroid, not a radius-of-gyration/RMS statistic. IID is not calculated. Milling is the absolute mean signed cross-product of unit centroid-relative positions and unit velocities: approximately 1 for a coherent mill and 0 for little net rotation. A coherent stationary-centroid mill can have high individual speed but near-zero centroid speed. Multiple counter-rotating vortices can cancel in a whole-group rotation score.

`plot` accepts processed results, ordinary simulation results, a run-folder path, or a pandas table. Use `table="positions"` for individual histories, `table="individuals"` for their time averages/traits, or `table="groups"` for processed physical-group metrics. With ordinary results it uses existing per-type summary columns, so new centroid/milling metrics require processing trajectories first. `time_bin=25` averages time plots into 25-unit bins of the plotted time coordinate, independently within repeats; omit it for raw values. Pass a Matplotlib `ax` to choose figure dimensions. `category` accepts `"rep"`, `"run"`, `"condition"`, `"agent_type"` (or `"type"`), any other table column, or a list. Processed condition labels include the parameter values.

Plot kinds are `"time"`, `"box"`, `"density"` and `"scatter"` (the latter needs `x=`). Time plots show separate repeat lines; tied largest-group values are averaged within each displayed type/time. Boxes and densities use one mean per repeat by default. Choose `unit="individual"` for distributions of individual averages, or explicitly `unit="frame"` for distributions of frame observations. Individuals sharing a group and consecutive frames are not independent experimental repeats. A constant/one-value density is displayed as a vertical line. All functions return editable Matplotlib axes; they do not add figure titles or save files automatically.

```python
plot(processed, "front_rank", kind="box", table="individuals",
     category="agent_type", unit="individual")
plot(processed, "milling", kind="scatter", x="polarisation", category="condition")
processed["individuals"].to_csv("individual_positions.csv", index=False)
```

Processing currently loads datasets into pandas memory, so large trajectory exports require sufficient RAM; it does not stream the post-processing itself. These are starter analyses and plots, not automatic interpretations. Tables can be exported to R for subsequent analysis. The new physical-group/positional metrics are calculated on demand after simulation; the existing live information panel and default simulation summaries retain their established lightweight measures.

## Trying Couzin-inspired milling interactively

The [Couzin et al. (2002) model](https://jmvidal.cse.sc.edu/library/couzin02a.pdf) has an omnidirectional repulsion zone and direction-limited orientation/attraction. CHIMESim now provides **Repulsion field of view (°)** in General movement settings, independent of the existing FOV slider. Set it to **360°**, while keeping the main FOV at **270°**. Repulsion still respects the detection-distance cap and intervening walls. The visual-neighbour overlay keeps using visual FOV; repulsion sensing is not automatically added to that graph. Interaction-zone diagrams show the independent repulsion sector. With no explicit repulsion FOV, scripted simulations retain the previous behaviour of using the main FOV. The interactive slider starts at that value and can then be changed independently.

This starting configuration maps the Fig. 4 parameters to ten model units per published length unit and one update per published 0.1-s timestep:

```python
milling = chimesim.settings(
    source="defaults", model="zonal",
    environment={"width": 1600, "height": 1000, "bounded": False, "spawn_radius": 70},
    movement={"desired_speed": 3, "fov_deg": 270, "repulsion_fov_deg": 360, "visual_range": 140},
    agents={"A1": {"n": 100, "repulsion": 10, "orientation": 20, "attraction": 140,
                   "max_turn_deg": 4, "noise": 0.05, "heading_noise_memory": 0,
                   "social_speed": 0, "speed_noise": 0}},
)
chimesim.interactive(settings=milling, seed=42)
```

Select **Lock arena size when resizing**, then Start. Explore orientation radii **15–35** and maximum turns **4–8°** separately and allow time for the group to organise; explore changes in both directions while retaining the evolving group. `spawn_radius` starts agents uniformly within a central circle; it is optional and capped to fit the arena, with wall cells excluded. It persists through GUI Reset and saved scenarios. This initial circle is an exploratory choice, not a calibrated reproduction of the paper's initial-condition distribution. Removing the key restores the usual dispersed initialisation.

A compact swarm with counter-rotating individuals is not necessarily coherent milling: look for low polarisation and sustained high milling within a physical group. At speed 3 and a 4° turn limit, the minimum steady-turn radius is approximately 43 model units; very tight clusters cannot form a coherent ring at that scale. Milling is not guaranteed for this preset or seed. CHIMESim remains a 2D adaptation, while the paper is 3D, and its open domain is only approximated here by a large periodic arena. Avoid interpreting boundary-wrap interactions as open-domain behaviour. The original paper's hysteresis requires parameter continuation within the same group; independent condition-grid runs reset the group and do not reproduce that protocol. No long milling experiment has been run automatically for this example.

[Calovi et al. (2014)](https://doi.org/10.1088/1367-2630/16/1/015026) instead studies a 2D data-driven **angular-velocity model**, derived from Gautrais et al.: constant forward speed, a stochastic turning-rate process, Voronoi neighbours, and summed alignment/attraction steering. Their added frontal weighting favours neighbours ahead and enables pronounced milling. This needs a distinct implementation; selecting CHIMESim's current `voronoi` mode or heading-noise memory does not reproduce those equations.

## Individual variation and agent parameters

Each agent-type window now has an **Individual variation** editor. Choose a trait, then:

- **Fixed**: all agents use the main slider value.
- **Normal**: the main slider sets the bell curve's centre, **Spread (SD)** its width, and optional minimum/maximum bound it. Draws come from a bounded normal distribution instead of being pushed onto the endpoints. With asymmetric bounds, the resulting mean can differ from the bell curve's centre.
- **Uniform**: enter a minimum and maximum; applying sets the main slider to their midpoint. Moving that slider shifts the range; near physical limits the symmetric range narrows.

**Apply variation** updates the selected trait for existing agents, keeping their positions, headings and other traits. Moving a main slider subsequently updates that trait's central value and resamples it; affected sliders are marked **mean**. Grey preview bars show current agents and the line shows the proposed distribution. Multiple traits can vary independently. Zone radii must remain ordered per individual; bounds are restricted by adjacent radii, and invalid changes leave the previous configuration intact.

The independent **Colour by** selector can show any supported trait, or restore **Type colour**. It does not resample values. A varying trait creates a gradient; a fixed trait gives one uniform colour. Colours share a scale across current agents of the same type, including newly added agents. **Speed multiplier** colouring shows that assigned trait, not instantaneous movement speed.

Saved settings retain distribution definitions and colour selection; starting or resetting a simulation samples new individuals. Existing Python configurations with lognormal/bimodal distributions remain supported; choosing Fixed, Normal or Uniform in the GUI replaces the selected distribution.

```python
scenario = chimesim.settings(source="defaults", agents={
    "A1": {"n": 100, "distributions": {
        "speed_scale": {"dist": "uniform", "low": 0.8, "high": 1.2},
        "r_a": {"dist": "normal", "mean": 75, "std": 10,
                "clip": [40, 110], "truncate": True},
    }},
})
results = chimesim.simulate(settings=scenario, n_steps=500, seed=42)
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
| `pursues`, `flees`, `captures` | Directed target-type names; capture/removal is separate and off by default |

Positions, radii and distances use pixels; speeds use pixels per model step. Python parameters use radians for `noise` and `max_turn`; the GUI displays degrees. `fov_deg` always uses degrees. Pursuit and fleeing share the social turn limit; wall steering is applied afterwards.

## Relationships between agent types

All types are general agent types; there are no built-in prey/predator roles. For example:

```python
relationships = chimesim.settings(source="defaults",
    movement={"visual_range": 200, "capture_distance": 10},
    agents={"A1": {"n": 100, "flees": ["A2"]},
            "A2": {"n": 5, "pursues": ["A1"], "captures": []}})
```

Pursuit alone never removes agents. Add `"captures": ["A1"]` to A2 to enable removal explicitly. Capture can also be configured without pursuit. The GUI exposes **Pursues**, **Flees** and **Captures** independently in each type's Relationships section, and the dataset setup GUI provides the same choices in Agents. Pursuing and fleeing the same target type are mutually exclusive. Relationships are directed: A2 pursuing A1 does not automatically make A1 flee. Unknown targets and self-targets are rejected.

Pursuit steers toward the centroid of perceived targets within the attraction radius. Fleeing steers away from the centroid of perceived threats within `flee_dist`. Both also respect visual range, its per-type multiplier, field of view, periodic distances and intervening walls. Fleeing takes priority over other steering; pursuit yields to same-school repulsion. Both pass through heading noise and the maximum-turn limit. The relationship rules are simple configurable steering, not a calibrated predator–prey model.

Capture uses `movement.capture_distance` (default 10 model units) and positions at the start of each step; it does not sweep movement paths. It requires a clear line of sight through walls, handles periodic seams and is independent of FOV or visual range. Set its distance to zero to disable it. All capture decisions use the shared starting snapshot; mutual capture can remove both agents. Removed individuals remain in `individuals.csv`, while their trajectories end and summary counts decrease.

Saved legacy `hunts` lists load as `pursues`; they no longer imply capture. Saved `catch_dist` loads as `capture_distance`. Newly saved settings use the new names. Legacy PREY/PREDATOR presets and duplicate scalar interaction helpers have been removed.

## Interaction models and environments

The **zonal** model gives repulsion priority; otherwise agents combine neighbour headings in the orientation zone and attraction toward neighbours in the outer zone. The **Voronoi** model restricts those rules to geometric neighbours, with `r_a` as the maximum interaction distance. Small populations remain interactive; degenerate triangulations fall back to zonal rules. Periodic arenas use tiled triangulation. Agent movement is calculated from a shared starting state and applied together.

Bounded arenas have outer walls; periodic arenas wrap at their edges. Metrics and directed interactions use the same boundary convention. Optional capture is evaluated at the starting snapshot, before movement; crossing paths within a step does not itself trigger capture.

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

Perception is opt-in and does not change movement or grouping. The GUI offers a visual-neighbour overlay. Advanced development can use internal implementation modules for a custom step-by-step experiment; these are outside the three-function interface and may change. For example:

```python
from chimesim import perception
from chimesim.agents import BOID
from chimesim.model import Model
from chimesim.environment import Environment

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

Triangle is the default; `body_style` can also be a per-agent list. Body size comes from `agent.type.size`. Display magnification and tail length do not affect perception; tails are not opaque bodies. Each observer looks along its movement heading, retaining its stored heading while stationary. `fov_deg` can be scalar or per-observer. All types can occlude one another regardless of directed relationships. No behavioural weights are inferred from visibility.

The calculation integrates target-centred angular rays, defaulting to at least 32 samples per in-FOV silhouette segment and a maximum angular spacing of 0.5 degrees. Increase `samples_per_body` or decrease `angular_resolution_deg` to resolve narrower occlusions. This is a numerical approximation, not an exact visibility polygon. Without occlusion, angular widths/FOV clipping are geometric; occlusion fractions converge with sampling resolution. Use `agent_occlusion=False` to inspect wall/FOV effects alone.

`max_distance` is an optional scalar or per-observer centre-distance cutoff for targets and body occluders; omitted means unlimited. Angular outputs outside it are zero. Use a cutoff for large snapshots; this API is not run automatically every simulation frame. Matrix outputs use O(N²) memory; temporary ray/body matrices are processed in batches. Walls retain the existing computational grid. Without an environment, coordinates are ordinary nonperiodic space. With a periodic environment, only the nearest image of each body is considered, appropriate for local perception rather than arena-spanning bodies/views. Equal-depth overlapping bodies do not occlude each other; overlapping eyes/bodies are degenerate scenes, not a contact model.

### Visual-neighbour overlay

In **Display**, select **Show visual neighbours (directed)**. Arrows run from observer to observed neighbour; darker links indicate a greater visible fraction. Faint grey stippled links run centroid-to-centroid, above geometric links and below agents; arrows indicate direction. **Visual range** in General movement settings caps perception and neighbour responses in movement. Each agent-type window has **Detection range ×**, multiplying that base; zero disables that type’s visual detection. **Threat detect range** remains the independent fleeing-distance parameter used by the current movement models. **Minimum visible fraction** filters displayed links. The overlay uses the selected triangle/circle body style, physical body size and global field of view, including wall and agent occlusion. It does not apply social-model or hunting/schooling filters. Display magnification and tails do not affect visibility.

Perception runs in a background snapshot calculation, at most twice per second; link membership can lag moving agents. Pausing lets the pending snapshot complete. Geometric and visual overlays can be toggled independently. Visual link membership and body occlusion do not yet change movement or spatial grouping; the range cutoff itself does affect movement.

### Display scale and frame rate

**Show scale bar and FPS** toggles both overlays: FPS at the top left and a single unlabelled 100-unit bar at the bottom left, with ticks at 0, 25, 50, 75 and 100. The bar follows zoom. **Triangle size / tail length ×** is visual only: triangle length matches circle diameter plus tail at the same setting, while the circle radius stays fixed.

## Interaction-zone panel

Enable **Display → Show interaction zones**, or use **Windows → Interaction zones**. The panel is a plain white diagram that automatically fits repulsion (red), alignment (green), attraction (blue) and a solid visual-range outline to the available space. All sectors respect field of view. Multiple active types appear as separate diagrams with their type colours. There are no diagram-specific controls or reference dots. The displayed agent uses the selected body shape and visual size setting. The diagram shows type-level radii, not every individual's sampled values.

All zone values are **outer radii from the stored agent position**, not diameters or distances from the body's edge. A repulsion radius of 21 therefore spans a diameter of 42. The triangle's rendered total length is `type.size × (2 + display_scale) / 3`, matching circle diameter plus tail length. For example, size 15 and display scale 3.4 give a drawn length of 27 model units. Display scaling changes neither interaction radii nor perception body geometry.

## Ordered interaction ranges and detection

The GUI enforces `0 ≤ repulsion ≤ orientation ≤ attraction ≤ visual range × type multiplier`. Slider bounds follow adjacent radii and update immediately. Decreasing detection clips affected type values, individual radii and distribution bounds to keep them valid, without resampling unrelated traits. Increasing detection does not restore clipped values; increase attraction first, then orientation and repulsion as needed. Radius distribution bounds also follow these constraints. Parameter/trait recording captures these changes.

Visual range now caps actual zonal/Voronoi social responses, social speed, hunting and fleeing as well as the diagnostic visual network. At zero effective range, agents have no neighbour-induced steering/acceleration; independent movement noise, preferred-speed recovery and walls still operate. The current movement models retain their existing point-to-point wall checks, not body-to-body occlusion from the separate perception module. Threat detection remains a separate distance with the effective visual range as an additional cap. Captures retain their existing contact-distance rule.

Headless `simulate(visual_range=...)` and `Model.step(..., visual_range=...)` accept the same cap. Omitted/None preserves unrestricted legacy detection for headless code; GUI settings save an explicit range, initially 200 for sessions without one.

## Heading noise and temporal persistence

**Heading noise (SD, °)** controls Gaussian error on the desired social direction before the maximum-turn limit. **Noise memory (steps)** defaults to zero, preserving independent errors and existing seeded trajectories. Positive values introduce exponentially correlated Gaussian heading errors: `error_next = exp(-1/memory) × error_previous + sqrt(1-exp(-2/memory)) × SD × normal(0,1)`. Initial errors use the stationary Gaussian distribution. The parameter is a correlation time in simulation steps, not seconds; ten steps means a lag-one correlation of about 0.905 before turn clipping. Setting noise SD to zero removes the error immediately.

This is an optional correlated-heading-error adaptation. It is not a reproduction of [Gautrais et al. (2012)](https://doi.org/10.1371/journal.pcbi.1002678), whose data-driven fish model puts an Ornstein–Uhlenbeck process on angular velocity/curvature. That distinction matters: persistence here reduces rapid random reversals, but social or wall steering can still change abruptly. [Couzin et al. (2002)](https://doi.org/10.1006/jtbi.2002.3065) uses independent wrapped-Gaussian desired-direction error followed by a turning-rate limit in 3D. The default CHIMESim rule retains that broad ordering in 2D without claiming exact model reproduction. There is no display-only heading smoothing.

## Advanced callbacks and implementation access

The Python API supports `simulate(progress=callback, should_cancel=callable, on_row=callback, collect_results=False)`. Progress receives completed/total steps, repeat, phase and elapsed time, plus condition information for grids. `on_row(kind, row)` receives `summary`, `trajectories`, `individuals` or `repeats` rows. Callbacks execute in the simulation thread and must not manipulate Tk widgets. With `output_dir=...`, streaming is managed by the package; progress and cancellation remain available, while `on_row` and `collect_results` should be omitted.

The former top-level `AgentType`, `Boid`, `BOID`, `PREY`, `PREDATOR`, `Model` and `Environment` exports have been removed. Existing advanced scripts can import the remaining classes and generic BOID preset from `chimesim.agents`, `chimesim.model` and `chimesim.environment` when step-by-step intervention or custom model development requires them. Ordinary configuration, heterogeneity, environments and repeats use the shared scenario interface instead. The former `run()` function has also been removed; use `interactive()` for the live visualiser.
