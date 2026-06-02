# Model overview
A modular, 2-D agent-based modelling framework. 
The goal for the finished model is to have the following functionalities:
- **Graphics mode and data collection mode** -- It can be run with a simple GUI or run without GUI, and instead collect specified data points over a specified number of simulation runs.
- **Can model environmental gradients** -- such as the depth of drying streams
- **Can model individual heterogeneity** -- such as boldness and social-proximity
- **Individual movement behaviours and interaction rules with realism**

<img width="1154" height="787" alt="image" src="https://github.com/user-attachments/assets/8f95c61b-8d95-46fb-83f2-832f2fecaa78" />

## Current status

**Working features:**
- Modular rule registry system with dynamic rule compilation
- Visualization via Tkinter GUI — canvas fills the screen, live FPS counter
- Three scenario toggles for controlling agent behaviour
- Boid flocking mechanics (centering, velocity matching, avoidance)
- Predator hunting and prey detection
- Live agent-count sliders (add or randomly remove agents while the simulation runs)
- Numpy-vectorised social rules (≥5× speedup over the Python baseline at typical counts)

**In progress / incomplete:**
- Data collection mode (non-GUI simulation runs)
- Environmental gradients (a basic radial refuge exists in the codebase but is disabled; more complex gradients not modelled)
- Individual heterogeneity (e.g. boldness, social-proximity)
- Signal/communication system for predators (partially coded, currently unused)

## Architecture overview

The model follows a simple pipeline:

1. Initialize `Model` with scenario toggles (`ENABLE_SOCIAL`, `ENABLE_CLASS_DIFF`, `ENABLE_PREDATION`)

	↓

2. `compile_rules()` examines toggles → selects rule groups → resolves to function objects

	↓

3. `step()` is called each frame:
	- Apply pre-compiled rule functions to each agent in sequence
	- Update positions and velocities
	- Check for predation events

	↓

4. Visualization (optional): Render agents and environment to canvas

The key insight: rules are decoupled from the step logic — change the registries or `compile_rules()` to alter behaviour.

## How it works
Upon running the model, the user can specify the number of agents (2 classes available, boids and predators) to be generated, in addition to the scenario under which the simulation is run. The model is designed to allow a variety of scenarios, with increasing complexity. In the current version, the scenario is set with 3 scenario toggles found in `config.py` -- `ENABLE_SOCIAL`, `ENABLE_CLASS_DIFF`, `ENABLE_PREDATION`.

**Scenario toggle definitions**

`ENABLE_SOCIAL`
- When disabled, there are no social interactions between agents. Each individual wanders randomly. 
- When enabled, agents start to move according to class-specific social interaction rules. For boids, these are: `avoid_others`, `match_velocity`, and `move_toward_center`, which are stored in `agent_rules.py`.

`ENABLE_CLASS_DIFF`
- When disabled, predators behave according to the same social rules as boids. 
- When enabled, predators get assigned their specific social rules, which include: `move_toward_center`, `pred_avoid_others`, these are similar to those for boids, with slightly different parameters. 

`ENABLE_PREDATION`
- When disabled, predators would not pursue the boids. There are no interspecific interactions. 
- When enabled, predators start to pursue and hunt the boids. Once the boids are within a specific distance, `CATCH_DIST` (stored in `config.py`) , of any predator, they get "eaten".

For a summary of the scenarios currently possible in this framework, see below:
![alt text](image.png)

## Rule registry and how to modify rule sets
The framework uses rule registries to automatically apply the correct behavioral rules to agents based on the three scenario toggles. Here's how it works at a technical level:

**Rule Registries**
Each agent class (boid and predator) has a rule registry stored in `functions/agent_rules.py` and `functions/predator_rules.py`. Each registry consists of two dictionaries:
- `rule_groups`: Maps group names to lists of function names (e.g., `'agent social rules': ['avoid_others', 'match_velocity', 'move_toward_center']`)
- `rule_names`: Maps function names to the actual function objects that implement those behaviors

**Compilation Process**
When the Model is initialized, the `compile_rules()` method in `model.py` examines the three scenario toggles (`ENABLE_SOCIAL`, `ENABLE_CLASS_DIFF`, `ENABLE_PREDATION`) and dynamically constructs a list of rule group names that should be active. For example, if `ENABLE_SOCIAL=True` and `ENABLE_PREDATION=True`, boids will be assigned `['pred_avoid_rules', 'agent_social_rules']`. Predators might be assigned `['predation_rules', 'pred_social_rules']`.

**Rule Resolution**
These rule group names are passed to `get_agent_rules()` or `get_pred_rules()` (depending on agent type), which resolves the group names by looking them up in the registries. If a name matches a key in `rule_groups`, it expands to the list of function names; if it matches a key directly in `rule_names`, it's used as-is. The result is a compiled list of tuples: `(function_name, function_object)`, stored in `self.active_boid_rules` and `self.active_pred_rules`.

**Runtime Execution**
During each simulation step, the `step()` method iterates through each agent and applies its pre-compiled rule list in sequence. This decouples rule definition from rule application, making it easy to add new rules or modify how rules are selected based on scenario toggles without changing the step logic.

**How to modify agent behaviour through changing rule sets** 

To modify agent behavior, you can make changes at different levels:

1. **Modify individual rule functions** — Edit the behavior of a specific rule (e.g., `avoid_others()`, `hunt()`) in `functions/agent_rules.py` or `functions/predator_rules.py`. Changes affect all scenarios that use that rule.

2. **Add a new rule function** — Write a new function in the appropriate rules file, then add it to the corresponding `rule_names` dictionary so it can be referenced.

3. **Create a new rule group** — Add a new key-value pair to `rule_groups` (e.g., `'bold_boids': ['match_velocity']`) to group multiple rules together.

4. **Modify which rules apply to each scenario** — Edit the `compile_rules()` method in `model.py` to change which rule group names are assigned for different combinations of the three toggles. For example, you could add a new rule group to boids when `ENABLE_SOCIAL=True`.

5. **Add new scenario toggles** — Extend the Model class in `model.py` to accept additional boolean parameters, then add conditional logic in `compile_rules()` to determine which rules activate based on the new toggles.

The advantage of this structure is that you can test behavioral variations without modifying the `step()` method or agent classes—only the rule registries and compilation logic need to change.

# File descriptions
**Core model files:**
- `agents.py` – Agent class definitions (Boid, Predator)
- `config.py` – Global parameters (scenario toggles, agent counts, behavioral constants)
- `environment.py` – Environmental objects (e.g., refuge, feeding patch)
- `model.py` – Core Model class that compiles rules and runs simulation steps
- `ui_tk.py` – Tkinter-based visualization

**Rule system:**
- `functions/agent_rules.py` – Boid behavior rules and rule registry (`agent_rule_groups`, `agent_rule_names`)
- `functions/predator_rules.py` – Predator behavior rules and rule registry (`predator_rule_groups`, `predator_rule_names`)
- `functions/` – Additional helper functions for agent interactions

**Tutorial / entry points:**
- `ABSim_tutorial.ipynb` – Jupyter notebook: model setup, headless runs, scenario comparison, GUI launch

**Reference scripts (temp/):**
- `temp/boid_predator_v3.py` – Original pre-modularisation single-file model (for reference)
- `temp/run_baselinemodel.py` – Minimal script to run the baseline model configuration

## Agent attributes and customization

**Boid properties:**
- **Position & velocity:** `x`, `y`, `dx`, `dy`
- **Movement:** `head` (heading angle), `speed`, `max_turn` (turning constraint)
- **Perception:** `fov` (field of view in radians, default 0.75π), `cohesion_visual_range` (how far agents can see neighbors)
- **Behavior:** `centering_factor` (strength of attraction to center of neighbors), `state` (currently "calm", extensible)
- **History:** `history` (list of past positions, for optional trail drawing)

**Predator properties (inherits from Boid):**
- All Boid properties, plus:
- **Hunting:** `move_state` ("wander" or hunting), `max_turn` (0.15π, more constrained than boids)
- **Communication (unused):** `signal_state`, `signal_timer`, `signal_pos`, `signal_response_strength`

**Per-agent customization:**
Currently, agent properties use class-level defaults from `agents.py` and `config.py`. To create individual heterogeneity (different boldness, perception ranges, etc.), modify the `__init__` methods in `agents.py` to accept per-agent parameters.

**Global customization:**
Edit parameters in `config.py` to change behavior across all agents of a class. Key parameters are listed below.

## Configuration parameters reference

**Scenario toggles** (in `config.py`):
- `ENABLE_SOCIAL` – Activate social interaction rules
- `ENABLE_CLASS_DIFF` – Use class-specific rules (predators differ from boids)
- `ENABLE_PREDATION` – Activate hunting and predation

**Canvas & simulation** (in `config.py`):
- `WIDTH`, `HEIGHT` – Canvas dimensions (pixels)
- `MARGIN` – Boundary margin for edge wrapping

**Boid parameters** (in `config.py`):
- `NUM_BOIDS` – Initial number of boids
- `BOID_VIS_RANGE` – Distance at which boids perceive neighbors (75 px default)
- `SPEED_LIMIT` – Maximum velocity magnitude
- `TURN_FACTOR` – Strength of wall-avoidance behavior
- `BOID_CENTERING_FACTOR`, `AVOID_FACTOR`, `MATCHING_FACTOR` – Weights for centering, avoidance, and velocity matching rules
- `DFLT_AVOID_DIST` – Minimum distance before avoidance triggers
- `DFLT_FLEE_DIST` – Distance at which boids detect and flee from predators
- `FLEE_FACTOR` – Strength of fleeing behavior

**Predator parameters** (in `config.py`):
- `NUM_PREDS` – Initial number of predators
- `HUNTING_SIGHT` – Distance at which predators detect prey
- `PRED_COHESION_VIS_RANGE` – Distance at which predators perceive other predators
- `PRED_CENTERING_FACTOR` – Weight for cohesion with other predators
- `DFLT_PRED_AVOID_DIST` – Distance for inter-predator avoidance
- `CATCH_DIST` – Distance at which predators catch and eat boids
- `PRED_REP_WANDER_FACTOR`, `PRED_REP_HUNT_FACTOR` – Weights for inter-predator repulsion

**Environment** (in `config.py`):
- `REFUGE_R`, `REFUGE_CX`, `REFUGE_CY` – Circular refuge radius and center position
- `REFUGE_REPEL` – Strength of refuge repulsion on agents
- `REFUGE_BUFFER` – Buffer zone around refuge

## Dependencies & setup

**Python packages** — installed automatically via `pip install -e .`:
- `numpy`
- `jupyter` / `matplotlib` — optional, for the tutorial notebook (`pip install jupyter matplotlib`)

**tkinter** is part of the Python standard library and cannot be pip-installed, but the version bundled with macOS is too old to render the canvas. See the macOS note below.

## Installation

### macOS (Homebrew + pyenv)

The macOS system Tk causes the simulation canvas to appear as a blank grey window. Use Homebrew's Python 3.9 with the proper Tk bindings instead:

```bash
# 1. Install Tk bindings for Homebrew Python 3.9
brew install python-tk@3.9

# 2. Clone the repo and enter it
git clone https://github.com/jollejolles/ABSim.git && cd ABSim

# 3. Create a virtual environment using the Homebrew Python directly
/usr/local/bin/python3.9 -m venv .absim
source .absim/bin/activate

# 4. Install the package (and optionally Jupyter + matplotlib)
pip install -e .
pip install jupyter matplotlib   # optional
```

Verify Tk is working:
```bash
python -c "import tkinter; r=tkinter.Tk(); print(r.tk.call('info','patchlevel')); r.destroy()"
# Should print 8.6.x or higher (not 8.5)
```

> **Note:** if `python3.9` on your PATH is shadowed by pyenv, use the full path `/usr/local/bin/python3.9` as shown above.

### Other platforms

```bash
git clone https://github.com/jollejolles/ABSim.git && cd ABSim
python -m venv .absim
source .absim/bin/activate   # Windows: .absim\Scripts\activate
pip install -e .
```

## How to run the model

### Jupyter notebook (recommended)

Open `ABSim_tutorial.ipynb` for a guided walkthrough: model initialisation, headless simulation runs, scenario comparisons, and launching the GUI.

```bash
source .absim/bin/activate
jupyter notebook ABSim_tutorial.ipynb
```

### GUI only

```python
from ABSim.ui_tk import run
run()
```
