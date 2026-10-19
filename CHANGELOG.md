# Roadmap and changelog

## Roadmap

### High priority

- **Force-based movement model:** add a distinct model based on Klamser et al. (2021), with continuous attraction/repulsion, velocity alignment, preferred-speed recovery, temporal fluctuations and coupled speed-dependent turning. Check equations and reference scenarios before claiming reproduction.
- **Visual networks, then vision-based movement:** the independent perception layer is implemented. Next add a directed visual-network overlay, focal-agent inspection and optional visual grouping. Then define behavioural responses to visible fraction/angular extent and add model controls.
- **Python support refresh:** migrate the development environment and verify Python 3.11+ before changing the supported-version requirement.
- **Parameter sweeps:** wrap `simulate()` to iterate over parameter grids, reuse GUI-saved settings and return clearly indexed results.

### Arena navigation and environmental extensions

- **Drawing tools:** extend the existing brush, rectangle and ellipse tools with polygons and selecting/editing geometry.
- **Traversable zones:** introduce cover/refuge and habitat zones with explicit, agent-type-specific effects on detection and movement, separate from solid walls.
- **Depth fields and responses:** draw a depth map and specify how local depth changes preferred speed, habitat preference or access, potentially differing among agent types.
- **Optional physical units:** consider explicit length/time units later; retain model coordinates for now.
- **Continuous gradients:** draw a line defining start/end environmental values; combine multiple lines into a reproducible 2D heterogeneous field.
- Save environmental definitions and edits with settings and recordings, and allow fields to be visualised independently of agents.

## Changelog

## [Unreleased]

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
