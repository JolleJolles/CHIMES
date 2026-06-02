# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Renamed Python package from `abmsim` to `ABSim`; all imports are now `from ABSim.xxx import …`
- Moved from `src/` layout to flat layout: package lives directly at `ABSim/` in the project root
- `pyproject.toml` rewritten: flat layout, `name = "ABSim"`, added `license`, `urls`, `[tool.setuptools.packages.find]`; removed redundant `setup.py`
### Removed
- `setup.py` (superseded by `pyproject.toml`)
- `temp/` directory (`boid_predator_v3.py`, `run_baselinemodel.py`) — nothing unimplemented remained
- `image.png` (broken reference in README)
- `.claude/settings.json` (development hook, not part of the package)
### Fixed
- Exit button now calls `root.destroy()` (stops the Tkinter event loop and closes the window) instead of `root.quit()` (which left the window open and the kernel blocked)
### Fixed
- `numpy` was missing from `pyproject.toml` dependencies; `pip install -e .` now installs it automatically
- "Apply above settings" button only updated internal config variables but never reinitialised agents; it now calls `reset_arena()` so the new boid/predator counts take effect immediately
- `draw()` was commented out in `reset_arena()`, leaving the canvas blank after every reset; agents are now rendered immediately so the initial state is visible before starting the simulation
- Canvas size (1920×1080) could exceed the screen width when combined with the control panel, pushing the canvas off-screen and leaving only a grey window visible; `run()` now creates the root window first, queries the screen dimensions, and caps `config.WIDTH`/`config.HEIGHT` so the window always fits
- Removed unused `WIDTH`, `HEIGHT`, `REFUGE_R`, `REFUGE_CX`, `REFUGE_CY` imports from `ui_tk.py` (all were from disabled features or replaced by `config.*` references)
- Added macOS Tk installation guidance to README and `pyproject.toml`: the macOS system Tk causes blank canvas rendering; confirmed working path is `brew install python-tk@3.9` + `/usr/local/bin/python3.9 -m venv`
- ◀ / ▶ nudge buttons flanking every slider so values can be incremented by one step at a time
- FPS counter displayed in the top-left corner of the canvas (exponential moving average, α=0.2, updated every frame)
- Removed hard 30 fps cap: simulation loop now uses `canvas.after(0, …)` to run as fast as possible instead of a fixed 33 ms delay
- Replaced O(n²) Python loops in `avoid_others`, `match_velocity`, and `move_toward_center` with a numpy-vectorised batch step (`Model._step_boids_vectorized`); the rule-registry system is preserved for the non-social (wander) fallback; measured speedups: ×5 at 60 agents, ×10 at 120, ×16 at 200
- N Boids and N Predators are now sliders (0–500 and 0–30) with live callbacks: dragging immediately adds new agents or removes a random selection from the current population; the "Apply above settings" button and text-entry boxes are removed
- Canvas now fills the full available window area instead of having a fixed 1920×1080 size; `run()` builds the UI skeleton first, calls `root.update()` to resolve geometry, then reads the actual canvas dimensions from `winfo_width/height` and writes them to `config.WIDTH`/`config.HEIGHT` before creating the model — so arena bounds always match what is visible
### Added
- `ABSim_tutorial.ipynb` — Jupyter notebook covering model initialisation, headless simulation, scenario exploration, parameter comparison, and GUI launch; replaces `temp/run_model_tk.py` as the primary entry point
### Removed
- `temp/run_model_tk.py` — superseded by the tutorial notebook
- `tests/` — empty placeholder folder
- `examples/` — empty placeholder folder
- `Demo_v.0.0.3.mp4` — demo video (large binary, not useful in repo)

## [0.0.3] - 2026-05-21
### Added
- Refactored the model into a modular framework. Three scenario toggles in `config.py` control which rules agents follow.
- Rule registries for boids and predators in `functions/agent_rules.py` and `functions/predator_rules.py`. Each registry has two dicts: one mapping group names to lists of function names, one mapping function names to function objects.
- `Model.compile_rules()` in `model.py` dynamically selects rule groups based on the three toggles, then resolves them to compiled `(name, func)` tuples stored in `self.active_boid_rules` and `self.active_pred_rules`.
### Changed
- Signaling functions disabled in the hunting scenario for simplicity.
### Fixed
- N/A

## [0.0.2] - 2026-03-23
### Added
- Exit button to terminate the application.
### Changed
- Rearranged widget order; widgets are now more clearly separated.
- Refuge geometry disabled (commented out) for now.
### Fixed
- N/A

## [0.0.1] - 2026-03-16
### Added
- Two sliders for `CATCH_DIST` (predator catch distance) and `SPEED_LIMIT`; values are passed live into `model.step()` via `loop()`.
- Two input boxes for customising number of boids and predators, with an apply button.
- Start, pause, and reset buttons for simulation control.

## [0.0.0] - 2026-02-14
### Added
- Installation and run instructions in README.
- Brief model description (WIP).
### Changed
- Moved original single-file model `boid_predator_v3.py` and run scripts (`run_baselinemodel.py`, `run_model_tk.py`) into `temp/`.

---

## Backlog / To-Do (Priority, Difficulty)

### GUI and interactivity
- Design a way for the user to specify scenario toggles upon running the interactive window, instead of manually changing them in `config.py` (High, Mid)

### Model structure and logic

### Model functionalities
- Write a function that runs headless simulation for a specified number of runs and collects basic per-agent data (High, Mid)

### Naming things
- Rename certain functions to make them more intuitive (Mid, -)
- Brainstorm short, accurate, and unique package names (Mid, -)
