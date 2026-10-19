# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Stabilisation fixes
- Apply arena distances and wall visibility consistently to hunting, fleeing and predation.
- Trace complete sight and movement segments through the wall grid, prevent tunnelling, preserve thin mask walls, spawn in free space, and rebuild wall geometry after GUI resizing.
- Retain Voronoi social interactions with two or three agents, and use zonal interactions for degenerate triangulations.
- Calculate group movement from a shared starting snapshot and apply next states together; resolve predation simultaneously and use stable group ordering.
- Load complete agent configurations from dictionaries, JSON and saved GUI sessions; explicit arguments override file values. Return reusable resolved settings, resolve relative mask paths, reject unknown settings, and use per-model random generators without changing global random state.
- Make headless settings opt-in, honour explicit bounded arguments, and report invalid or missing settings files.
- Import the GUI only when run() is called, so headless installations do not require Tkinter.
- Compute nearest-neighbour distances using the actual arena boundaries in both GUI and headless output.

### Added
- **Wall / mask physics** (`ABSim/environment.py`): arena boundary repulsion (analytical O(N)) and optional interior walls loaded from an image file (dark = wall, white = free). Per-agent forces computed from a downsampled SDF grid (scipy EDT or pure-numpy L1 fallback).
- **Pure-numpy SDF fallback** (`_sdf_numpy`): 4-sweep L1 distance transform so wall physics work even when scipy is not installed.
- **Line-of-sight (LOS) check**: midpoint test on the downsampled wall grid blocks perception through walls; skipped entirely (returns `None`) when no mask is loaded.
- **Headless runner** (`ABSim/simulate.py`): `simulate()` function for parameter-sweep style batch runs. Supports `n_steps`, `n_reps`, `burn_in`, optional trajectory saving, reproducible seeding, and settings files merged on top of the auto-save.
- **Metrics module** (`ABSim/metrics.py`): `polarisation()` (mean-resultant length), `mean_nnd()` (nearest-neighbour distance), and `compute()` returning a flat dict of per-group and overall stats.
- **Individual heterogeneity**: `AgentType.set_distribution(param, dist, …)` registers a sampling distribution (normal / uniform / lognormal / bimodal) for any numeric parameter. On reset, each agent receives a sampled value in its `Boid.individual` dict. Per-agent arrays are built once per tick in `_step_group` for vectorised operation.
- **Colormap-based agent colouring**: first `set_distribution()` call auto-enables `color_by`; agents are coloured on a matplotlib colormap (viridis, plasma, etc.) or a blue→red fallback. A 48-cell gradient colorbar is pre-computed on `assign_individual()`.
- **Colorbar HUD**: right-side canvas overlay shows a gradient bar + min/max labels for each agent type that has `color_by` set.
- **Parameter name aliases** (`_PARAM_ALIASES`, `_resolve_param()`): friendly names (`"speed"`, `"zor"`, `"zoo"`, `"zoa"`, `"detect"`, `"maxturn"`) accepted by `set_distribution()` and mapped to internal attribute names.
- **`AgentType.clone(name=None)`**: independent deepcopy with optional rename, replacing the pattern `copy.deepcopy(BOID)`.
- **Dynamic agent type count** (1–3) in GUI: ◀/▶ buttons add/remove agent-type panels live. New types copy all behaviour parameters from the last existing type.
- **Auto-save/load settings**: slider values, arena options, and per-type parameters are written to `~/.absim/settings.json` on exit and restored on next launch.
- **Scrollable agent-type panel**: right sidebar has a scrollable region so multiple agent panels fit on any screen height.
- **Wall mask overlay**: interior walls rendered as canvas rectangles (row-run-length encoding) rather than a full-resolution PhotoImage — compatible with Retina/HiDPI displays at full FPS.
- **Resizable sidebar**: left canvas and right control panel separated by a draggable `tk.PanedWindow` sash; sliders are fluid-width (fill the available panel width).
- **Steps-per-frame control**: ×1 ×5 ×20 ×100 buttons in the top panel; the simulation runs the selected number of model steps per rendered frame, decoupling simulation speed from display rate.
- **Thigmotaxis (wall-following)**: tangential force alongside normal wall repulsion causes agents to follow walls rather than bounce off them. Strength controlled by `Environment.thigmotaxis` (0 = off, higher = stronger following). Exposed as a "Wall follow" slider in the Arena panel and saved in settings.
- **Gaussian noise**: heading perturbation now sampled from N(0, σ) instead of Uniform(−η, η), matching the SDE literature. Default noise values updated to preserve equivalent spread (σ ≈ η√3). GUI label updated to "Noise σ (°)".
- **Multi-model architecture**: `AgentType.interaction_model` selects the social interaction model (`"zonal"` or `"voronoi"`). `Model._step_group` dispatches to `_social_zonal()` or `_social_voronoi()` after shared setup (distances, FOV, LOS). Social speed is a shared add-on layer applied after whichever model runs.
- **Voronoi interaction model**: `_social_voronoi()` builds a Delaunay neighbour graph (equivalent to Voronoi ridges) using `scipy.spatial.Delaunay`, applies the same repulsion/orientation/attraction rules as the zonal model but only to topological neighbours, with a max-interaction-distance cutoff. Falls back to zonal if scipy is unavailable.
- **Model-type dropdown in GUI**: each agent-type panel has an OptionMenu (`zonal` / `voronoi`). Switching the dropdown immediately updates `AgentType.interaction_model` and relabels the ZOR/ZOO/ZOA sliders to their Voronoi equivalents (Repulsion threshold / Orientation distance / Max interaction dist). Selection saved in per-type settings.
- ◀ / ▶ nudge buttons flanking every slider for single-step adjustment.
- FPS counter in top-left of canvas (exponential moving average α = 0.2).
- Per-group HUD stats: N, polarisation, NND shown in the agent type's colour.

### Changed
- Simulation loop changed from `canvas.after(16, loop)` (≈ 30 fps cap) to `root.after(0, loop)` + `root.update_idletasks()` — runs as fast as the display allows, typically 60–200+ fps.
- `_step_group` now builds per-agent parameter arrays at the start of each tick so individual variation (speed scale, zone radii, max turn, noise) is applied vectorised. Type-level values are the fallback when no distributions are set.
- Toroidal correction in `_step_group` is only applied when the arena is non-bounded; bounded arenas no longer incorrectly treat agents at opposite edges as neighbours.
- `Model.__init__` now accepts `agent_types` as a list (not just the built-in PREY/PREDATOR defaults).
- `ABSim/__init__.py` exports: `AgentType`, `Boid`, `BOID`, `PREY`, `PREDATOR`, `Model`, `Environment`, `run`, `simulate`, `metrics`.
- Renamed Python package from `abmsim` to `ABSim`; all imports are now `from ABSim.xxx import …`
- Moved from `src/` layout to flat layout: package lives directly at `ABSim/` in the project root.
- `pyproject.toml` rewritten: flat layout, `name = "ABSim"`, added `license`, `urls`, `[tool.setuptools.packages.find]`; removed redundant `setup.py`.
- Canvas now fills the full available window area; arena bounds match the visible canvas at launch and after resize.
- N agents and N predators changed to live sliders (0–500 / 0–30) that add/remove agents immediately.

### Fixed
- NaN crash on canvas resize: guard `new_w <= 1 or new_h <= 1` skips zero-size events during Tk layout.
- Agents piling at canvas edges after resize: positions are rescaled proportionally (`a.x *= new_w / old_w`).
- Fish swimming through walls when scipy is absent: pure-numpy `_sdf_numpy()` fallback ensures the SDF is always computed.
- FPS stuck at ~20 fps on macOS: `root.after(0, …)` instead of `canvas.after(…)` bypasses timer coalescing.
- Nothing drawn until Pause was clicked: `root.update_idletasks()` flushes pending draw calls before re-scheduling the loop.
- `TclError` on exit: `try/except TclError` guard around `root.destroy()`, and `root.after_cancel` (not `canvas.after_cancel`).
- `RuntimeWarning: invalid value in divide` in `_step_group`: `safe_d` now guards both `dists == 0` and `np.isinf(dists)`.
- `~/Desktop/mask.png` not found: `os.path.expanduser()` applied in `Environment.load_mask()`.
- `NameError: name 'BOID' is not defined`: `BOID` is now exported from `ABSim/__init__.py`.
- Wrong parameter name `"speed"` rejected by distributions: `_PARAM_ALIASES` maps `"speed"` → `"speed_scale"`.
- No colorbar shown until Start clicked: `color_by` defaults to `None` (auto-enabled on first `set_distribution()`) instead of `False`.
- `setup.py` removed (superseded by `pyproject.toml`).
- `numpy` added to `pyproject.toml` dependencies.
- Exit button now calls `root.destroy()` instead of `root.quit()`.
- `draw()` re-enabled in `reset_arena()` so the initial state is visible before Start is clicked.

### Removed
- `setup.py` (superseded by `pyproject.toml`)
- `temp/run_model_tk.py` (superseded by tutorial notebook)
- `tests/`, `examples/` (empty placeholder folders)
- `Demo_v.0.0.3.mp4` (large binary)

---

## [0.0.3] - 2026-05-21
### Added
- Refactored the model into a modular framework. Three scenario toggles in `config.py` control which rules agents follow.
- Rule registries for boids and predators in `functions/agent_rules.py` and `functions/predator_rules.py`.
- `Model.compile_rules()` dynamically selects rule groups based on the three toggles.
### Changed
- Signaling functions disabled in the hunting scenario for simplicity.

## [0.0.2] - 2026-03-23
### Added
- Exit button to terminate the application.
### Changed
- Rearranged widget order; widgets are now more clearly separated.
- Refuge geometry disabled (commented out) for now.

## [0.0.1] - 2026-03-16
### Added
- Two sliders for `CATCH_DIST` and `SPEED_LIMIT`; values are passed live into `model.step()` via `loop()`.
- Two input boxes for customising number of boids and predators, with an apply button.
- Start, pause, and reset buttons for simulation control.

## [0.0.0] - 2026-02-14
### Added
- Installation and run instructions in README.
- Brief model description (WIP).
### Changed
- Moved original single-file model `boid_predator_v3.py` and run scripts into `temp/`.

---

## Roadmap / To-Do

### High priority
- **Physics / force model** (`interaction_model = "force"`): second-order (Newtonian) social forces — spring-like attraction–repulsion (k_rep, k_att, d0), velocity-alignment force (μ), friction–propulsion with speed relaxation time (τ). Matches Puy et al. (2024, PNAS) and the broader SDE / active-matter literature. Parameters are measurable from empirical trajectory data. GUI: force-constant sliders replace zone-radius sliders when this model is selected.
- **Retinal-angle neighbour weighting**: scale social influence by the angle a neighbour subtends on the agent's retina (proportional to body size / distance²). Provides a more biologically realistic alternative to both metric radius and Voronoi; adds one parameter (minimum retinal-angle threshold below which a neighbour is ignored). Compatible with all interaction models.
- **Parameter sweep CLI**: `simulate()` wrapper that iterates over a parameter grid and returns a tidy DataFrame; integrates with the settings file so GUI-tuned presets carry over.
- **Wall mask editor**: draw/erase walls interactively inside the GUI canvas (flood-fill + brush tool) and save as PNG.
- **Bounded-arena thigmotaxis**: extend the tangential wall-following force to axis-aligned canvas edges (currently only applied for mask walls).
- **Replay / recording**: save a simulation run as a sequence of frames and replay it frame-by-frame; export as video or animated GIF.
- **User-defined scenarios in GUI**: replace `config.py` scenario toggles with checkboxes/dropdowns in the control panel.

### Medium priority
- **Spatial statistics overlay**: real-time density heatmap or Voronoi diagram rendered on the canvas, toggled via a checkbox.
- **Interaction-network snapshot**: compute and visualise the momentary social-network graph (who is in whose ZOO/ZOA) for a selected frame.
- **Obstacle primitives**: place circular or rectangular solid obstacles in the arena without requiring a full mask image.
- **Export data button**: one-click CSV/JSON export of the last N seconds of metrics from the GUI session.
- **Per-agent trail rendering**: optional ghost-trail behind each agent for visualising recent paths (configurable trail length).

### Lower priority / exploratory
- **Adaptive timestep**: variable Δt based on local density, keeping per-step displacements below one ZOR radius.
- **Asymmetric zones**: allow ZOR/ZOO/ZOA to be elliptical (forward-weighted) rather than circular.
- **Social force model option**: toggle between Couzin zone-based rules and a continuous-force (Helbing-style) interaction.
- **3-D simulation**: extend positions and headings to 3-D; render with a projected 2-D canvas view.
- **Plugin system**: load external `.py` rule modules at runtime so custom behaviours can be added without modifying the core package.
- **Rename package** to something short, accurate, and unique (current name `ABSim` is generic).
