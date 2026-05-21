# Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [To-Do List] (Priority, Difficulty)

### GUI and interactivity
- Design a way for the user to specify scenario toggles upon running the interactive window, instead of manually changing them in config.py (High, Mid)

### Model structure and logic

### Model functionalities
- Write a function that runs headless simulation for specified number of runs and collects basic data of each agent (High, Mid)

### Naming things
- Rename certain functions to make them more intuitive (Mid, )
- Brainstorm short, accurate and unique package names (Mid, )

## [Unreleased]
### Added
### Changed
### Fixed

## [0.0.3] - 2026-05-21
### Added
- Refractored the model so that it is more like a modular framework. In `config.py`, three scenario toggles control what rules the agents follow, based on the scenario.
- 'Rule registries' are created for boids and predators (in functions/agent_rules.py and functions/predator_rules.py). Each registry contains 2 dictionaries. One dictionary creates function groups (e.g., `'agent social rules': ['avoid_others', 'match_velocity', 'move_toward_center']`), the other maps function names to the functions themselves.
- `Model.compile_rules()` in `model.py` is created to dynamically select the function groups based on the three scenario toggles. Then the the list of functions to be activated gets passed to `get_agent_rules()`, which looks up the actual functions from the registries. 
- The result is a compiled list of tuples: `(function_name, function_object)`, stored in `self.active_boid_rules` and `self.active_pred_rules`.
### Changed
- For simplicity, signaling functions are disabled in the scenario where hunting is enabled.
### Fixed
- N/A

## [0.0.2] - 2026-03-23
### Added
- Added an exit button to terminate the application
### Changed
- Rearranged the order of widgets and separate the widgets more clearly
- Disabled refuge geometry for now 
### Fixed
- N/A

## [0.0.1] - 2026-03-16
- Added more interaction options in the interface

### Added
- Added 2 more sliders for interactive interface, for the global variables `CATCH_DIST` (the distance between a predator and a boid below which the boid is considered "eaten") and `SPEED_LIMIT` (speed limit for boids, although currently it affects boids and predators differently, which should be examined later) | The main changes are located in ui_tk.py, where tk variables are defined and passed onto `loop()` which contains `model.step()`, the main update function for the model
- Added 2 input boxes for customizing numbers of boids and predators to be initiated, and a button to apply the newly entered values.
- Added buttons for starting and stopping the simulation, and for resetting the arena. 

## [0.0.0] - 2026-02-14
- Model now functions the same as the previous single-file model

### Added
- Instruction on installation and running the model in README
- Brief description of the model (WIP)

### Changed
- Moved original single-file model `boid_predator_v3.py`, and 2 scripts that run the model, namely `run_baselinemodel.py` and `run_model_tk.py` into the temp/ folder.