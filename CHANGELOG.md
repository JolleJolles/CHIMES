# Roadmap and changelog

## Roadmap

### High priority

- **Force-based movement model:** add a distinct model based on Klamser et al. (2021), with continuous attraction/repulsion, velocity alignment, preferred-speed recovery, temporal fluctuations and coupled speed-dependent turning. Check equations and reference scenarios before claiming reproduction.
- **Visual networks, then vision-based movement:** the independent perception layer is implemented. Next add a directed visual-network overlay, focal-agent inspection and optional visual grouping. Then define behavioural responses to visible fraction/angular extent and add model controls.
- **Python support refresh:** migrate the development environment and verify Python 3.11+ before changing the supported-version requirement.
- **Data-driven angular-velocity model:** consider Gautrais/Calovi steering dynamics, Voronoi interactions and frontal weighting as a distinct model alongside the planned force-based model.

### Arena navigation and environmental extensions

- **Drawing tools:** extend the existing brush, rectangle and ellipse tools with polygons and selecting/editing geometry.
- **Traversable zones:** introduce cover/refuge and habitat zones with explicit, agent-type-specific effects on detection and movement, separate from solid walls.
- **Depth fields and responses:** draw a depth map and specify how local depth changes preferred speed, habitat preference or access, potentially differing among agent types.
- **Optional physical units:** consider explicit length/time units later; retain model coordinates for now.
- **Continuous gradients:** draw a line defining start/end environmental values; combine multiple lines into a reproducible 2D heterogeneous field.
- Save environmental definitions and edits with settings and recordings, and allow fields to be visualised independently of agents.

## Changelog

## [Unreleased]

- Use CHIMES as the public project name in the README, guides, tutorial and GUI, with chimesim as the distribution/import name.

- Add three compressed README screenshots, correct the acronym expansion to Collective, simplify compatibility information and update the creator/maintainer biography.

- Embed interaction-zone diagrams at the top of each type's settings window, retain independent visual-range toggles and global diagram visibility, cap example agents at 50 screen pixels while keeping zone proportions, and remove the separate zone window.
- Update the tutorial with the supplied predator/prey movement settings and a broad central spawn; retain explicit assumptions for counts, pursuit/fleeing and optional capture.

- Export a mouse-predator trajectory companion CSV with sample/step alignment, activity, position, travel heading and model-step velocity; keep inactive gaps explicit and reset heading/velocity across gaps or recording restarts.

- Add an optional mouse predator with a red cursor marker, zoom/pan-aware coordinates, wall-drawing exclusion and recorded position changes; reuse directed threat perception without adding agents or capture.

- Add a saved per-type body-size multiplier in both GUIs and scripted settings; share it across rendering, zone examples and visual occlusion, refresh paused visual networks on changes, and keep movement/collision/capture rules independent of body size.

- Replace built-in prey/predator assumptions with directed pursuit, fleeing and opt-in capture; expose the choices in both GUIs and save/log explicit relationships.
- Batch between-type perception, respect FOV/range/walls/periodic seams and the heading turn limit, validate relationship targets and remove duplicate scalar interaction rules.
- Rename catch distance to capture distance; load legacy hunts as pursuit without automatic removal, and retain old saved distance keys as input aliases.
- Document the observed compact-initialisation and gradual-alignment sequence for interactive milling.

- Rework the tutorial around GUI-first dataset generation, preserve independent scenario cells, preview result tables, show grid row counts, compare repeats and clarify exploratory milling.
- Offer labelled condition-parameter selectors, accept comma-separated values, add Stop simulation and Exit controls, and return the latest saved-run information after closing the setup GUI for loading/plotting.
- Add optional per-repeat time-bin averaging and thinner time-series lines; document frame distributions versus repeat-level comparisons.

- Flush native window-close events before `interactive()` returns to a notebook kernel, and ensure window destruction still runs if recording or visual-network cleanup raises.

- Repair invalid tiny saved window sizes, preserve hidden panels' last usable geometry, and reliably show/raise Information with its checkbox or menu command.
- Replace live GUI native option menus with read-only selectors, defer selection callbacks until popup closure, and avoid panel redraw/resize work while input is grabbed.
- Add a live 0–1 milling plot and per-type M readout, calculated within physical spatial groups before member-weighted averaging so separately rotating groups do not cancel; share rotation calculations with offline metrics and reuse the group-count sampling cadence.

- Add an independent universal repulsion field of view to zonal/Voronoi movement, GUI sliders, interaction-zone diagrams, saved settings and recordings; retain wall/range occlusion and legacy scripted defaults.
- Add optional circular compact initialisation for exploratory milling scenarios, preserving free-space spawning and saved/reset behaviour; document a Couzin-inspired 2D configuration and remaining reproduction differences.
- Add optional post-processing for physical-group milling, centroid distance/speed, NND, speed variation and shape, with largest-group selection including ties and per-individual time-averaged front/centre ranks.
- Add editable starter time-series, box, density and scatter plots with run/condition/type categories, optional analysis dependencies and repeat-level aggregation; keep trait relationships optional and omit all-pairs IID.
- Include optional analysis dependencies in automated compatibility checks.
- Focus the interactive spatial-position history on the largest physical groups; retain unavailable scores for ambiguous or poorly aligned groups.

- Clarify `burn_in` and add the preferred `output_dir` keyword for unique per-call run folders; retain `output` as an alias and explicit in-memory results when no directory is supplied.
- Export tracking-style trajectories with set/time/identity, positions, model velocities and radian headings; move effective individual trait assignments to a separate individuals table.
- Calculate online post-burn-in time means per repeat, preserve partial-repeat counts/cancellation, and save a versioned data-format description with streamed output; document R/swaRmverse input mapping and periodic-coordinate handling.

- Add optional `simulate(performance=True)` workload reports with wall/CPU timings, steps per minute, agent updates per second, per-condition/repeat detail and computer/software information; save reports with streamed output and offer the option in the setup GUI.

- Introduce a shared `settings()` scenario builder alongside `interactive()` and `simulate()`; remove legacy agent/model/environment classes and presets from top-level exports while retaining internal implementations.
- Reuse the last interactive scientific scenario by default, separate its persistence from window/display preferences, and retain loading of older flat settings files.
- Add validated Cartesian condition grids to dataset generation and its setup GUI, with condition/repeat identifiers, explicit resolved configurations and actual random seeds; support direct streaming output to run folders.
- Update the README, user guide and tutorial to the shared scenario interface.
- Collect closed Tk sessions on the main thread before launching simulation workers, preventing delayed Tk-variable cleanup in a worker.


### Clear public entry points
- Replace `chimesim.run()` with `chimesim.interactive()` without a compatibility alias; update tutorial examples and documentation.
- Add `simulate(gui=True, ...)` to open the separate setup/progress window with settings and explicit argument overrides prefilled; direct calls continue returning datasets.
- Rename the standalone launcher to `chimesim-simulate`, remove batch-labelled modules/entry points and document GUI disk output versus direct return values.

### Standalone experiment runner
- Add a separate setup/progress GUI with settings loading/saving, basic agent/arena/movement controls, repeats, burn-in, seed, output selection and cancellation.
- Reuse the shared simulation engine with progress, cancellation and streamed output callbacks, preserving seeded API behaviour and avoiding in-memory trajectory accumulation.
- Save unique run folders with resolved settings, summary/trajectory CSVs and completion/cancellation/failure metadata; save the generated seed when randomised.
- Add `simulate(gui=True)` and the `chimesim-simulate` entry point without importing Tk for headless use.

### Ordered ranges, simplified diagrams and heading persistence
- Simplify interaction-zone display to automatically fitted white diagrams, with no selectors, controls, reference dots or explanatory text; use FOV-limited solid visual-range boundaries.
- Dynamically constrain zone sliders to neighbouring radii and effective detection range; clip affected individuals/distribution bounds when limits shrink, avoiding invalid combinations from normal slider use.
- Apply global visual range and per-type multipliers to movement responses as well as perception; allow zero range and preserve the cap in headless configurations and parameter logs.
- Add optional exponentially correlated Gaussian heading errors through Noise memory (steps), keeping independent Gaussian errors as the default; document the distinction from angular-velocity OU models.

### Python and platform compatibility checks
- Add model-test matrices for Python 3.11–3.14 on Windows/Linux/macOS, a legacy 3.9 check and an exploratory 3.15 check.
- Add opt-in real-Tk smoke tests for recording/export, camera, visual-network display, reopening and shutdown; configure all three platforms with Linux Xvfb.
- Centralise native mouse-wheel bindings so X11 wheel-button events are registered only on X11.
- Record successful macOS Sequoia GUI checks on Python 3.9/3.11/3.12, document verification limits and pending remote checks, and preserve the existing environment and declared Python minimum.
- Align slider nudge arrows with the slider track by placing the numeric value in a separate row.

## [0.2.0] - 2026-10-19

### Models, individuality and metrics

- Add preferred-speed recovery, social acceleration/braking, temporal speed noise and speed ceilings to zonal/Voronoi movement; keep a full force-based model as future work.
- Support targeted individual-trait updates, bounded normal sampling, fixed/normal/uniform variation and reliable ordered radius sampling without changing unrelated traits.
- Centralise periodic-aware group metrics, adaptive spatial grouping, wall-aware connectivity and within-group front/back and centroid-distance measures.
- Add independent body-aware perception with FOV, partial wall/body occlusion, visible angular extent/fraction and observer-specific ranges; movement rules do not yet use it.
- Add reusable body geometry, serialisable drawn/filled wall definitions, outer-wall following and free-space spawning; filter wall checks to relevant pairs and compact ray traversal.

### Interactive exploration and environments

- Add trait-distribution editors/previews, trait colours, advanced speed controls and individual-position histories ordered by traits.
- Reorganise controls and A1–A3 panels, add live group plots and trails, and fix window shutdown and foreground behaviour.
- Add explicit recording start/stop/export, actual-trait and parameter-change audit files, plus validated settings loading.
- Add mask loading/removal, brush/rectangle/ellipse drawing, reversible fills and near-loop closure, with smooth cached display contours kept separate from the computational wall grid.
- Add zoom/pan/scrollbars, explicit arena locking, correct unlocked resize fit and triangle/circle-with-tail display controls.
- Add an asynchronous directed visual-neighbour overlay, range/multiplier controls and paused refresh, plus a shared scale-bar/FPS toggle.

### Documentation and checks

- Shorten the README, move detailed guidance into docs/usage.md and consolidate the roadmap here.
- Add focused tests for speed, heterogeneity, position metrics, walls and perception; check GUI recording, masks, camera, asynchronous links and shutdown separately.

## [0.1.0] — CHIMESim foundation

Development baseline recorded in the existing Git history; no separate GitHub release is implied.

- Rename ABSim to CHIMESim, retain contribution history and update maintainer/licence metadata.
- Stabilise configurations, seeded headless batches, synchronous updates, periodic geometry, wall collision/visibility and optional Tkinter imports.
- Introduce separate simulation, controls, information and agent-type windows, persisted layouts, arena sizing and trajectory/settings export.

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
