# CHIMESim

**Creating Heterogeneous, Interactive Movement and Environment Simulations.**

Interactive 2-D simulations of collective animal movement, with individual variation, multiple agent types and a headless Python API. Explore how changing movement and social parameters affects a group, then run the same configuration in repeated simulations.

The package is under development. It currently supports zone-based and Voronoi-neighbour interactions, bounded or periodic arenas, mask walls, and hunting/fleeing relationships. Its zone rules are inspired by Couzin-style models; it is not a validated reproduction of every detail of a particular published model.

<img width="1154" height="787" alt="CHIMESim screenshot" src="https://github.com/user-attachments/assets/8f95c61b-8d95-46fb-83f2-832f2fecaa78" />

## Install

From a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e .
```

NumPy and SciPy are installed automatically. Optional extras:

```bash
python -m pip install -e '.[masks]'      # Pillow, for wall-mask images
python -m pip install -e '.[notebook]'   # Jupyter, matplotlib and pandas
```

Headless simulations do not require Tkinter. To use the interactive window, install Python with working Tkinter bindings and verify them with `python -m tkinter`. The Tk bindings must match the Python interpreter in your environment.

The declared minimum remains Python 3.9 for this stabilisation release. Updating and validating the supported Python versions is on the roadmap.

## Interactive exploration

```python
from chimesim import run

run()
```

The GUI restores its last session from `~/.chimesim/settings.json`, falling back to previous ABSim settings on first launch. It supports up to three agent types, live movement controls, population changes, a model selector, group metrics, individual-value colours and a geometric-neighbour overlay. The overlay shows Delaunay geometry, not the complete directed interaction network after zone, visibility and relationship filters.

Use explicit agent types to start a custom scenario:

```python
from chimesim import BOID, run

slow = BOID.clone("slow")
fast = BOID.clone("fast")
slow.speed_scale = 0.8
fast.speed_scale = 1.3
run(agent_types=[slow, fast], bounded=True, seed=42)
```

Types without hunting/fleeing relationships school together. Changing the steps-per-frame control advances more model steps between drawings. `run(log=True)` returns metrics for each model step after the window closes; resetting clears that session log. Resizing changes arena dimensions and rescales positions, so a live GUI session is exploratory rather than a fixed experiment.

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

The loader also accepts GUI names `speed`, `fov`, `n`, `max_turn_deg` and `noise_deg`. `run(settings="experiment.json")` can open the agent and movement configuration in the GUI; the interactive arena follows the window size. Run length and repeats apply to headless runs.

## Individual variation and agent parameters

```python
from chimesim import BOID, simulate

fish = BOID.clone("fish")
fish.set_distribution("speed", dist="uniform", low=0.8, high=1.2)
fish.set_distribution("zoa", dist="bimodal", means=[60, 100], stds=[5, 5],
                      clip=(35, 150))
results = simulate(agent_types=[fish], n_steps=500, seed=42)
```

Distributions support normal, uniform, lognormal and bimodal sampling. Supported individual parameters are `speed_scale`, `r_r`, `r_o`, `r_a`, `flee_dist`, `max_turn`, `noise` and `social_speed`. Aliases include `speed`, `zor`, `zoo`, `zoa`, `detect`, `maxturn` and `socspeed`.

Samples are clipped to nonnegative physical values; `social_speed` is also limited to 1 and `max_turn` to π. Sampled radii must satisfy `r_r <= r_o <= r_a`; use suitable spreads and `clip` bounds. Individual values remain fixed for an agent's lifetime and take precedence over type-level values. Resetting samples new individuals. Colours are normalised over each newly sampled cohort.

| Parameter | Meaning |
|---|---|
| `n_default` | Initial agent count |
| `r_r`, `r_o`, `r_a` | Ordered repulsion, orientation and attraction outer radii |
| `speed_scale` | Multiplier on `desired_speed` |
| `max_turn` | Maximum social heading update per step, radians |
| `noise` | Gaussian heading-noise standard deviation, radians; applied before turn clipping |
| `social_speed` | Speed adjustment strength, 0–1; zero gives fixed preferred speed |
| `flee_dist` | Threat detection distance |
| `interaction_model` | `"zonal"` or `"voronoi"` |
| `hunts`, `flees` | Names of other types to hunt or flee |

Positions, radii and distances use pixels; speeds use pixels per model step. Python parameters use radians for `noise` and `max_turn`; the GUI displays degrees. `fov_deg` always uses degrees. Wall and predator steering are applied after the social turn limit.

## Interaction models and environments

The **zonal** model gives repulsion priority; otherwise agents combine neighbour headings in the orientation zone and attraction toward neighbours in the outer zone. The **Voronoi** model restricts those rules to geometric neighbours, with `r_a` as the maximum interaction distance. Small populations remain interactive; degenerate triangulations fall back to zonal rules. Periodic arenas use tiled triangulation. Agent movement is calculated from a shared starting state and applied together.

Bounded arenas have outer walls; periodic arenas wrap at their edges. Metrics and hunting/fleeing use the same boundary convention. Predation is evaluated before movement, so a capture between two recorded positions is not detected until the next step.

For image walls, install the `masks` extra and provide `mask="arena.png"`. Dark source pixels become solid grid cells. Walls use an approximately four-pixel grid, so thin walls are preserved but may be widened. Visibility checks traverse the complete segment through that grid, and movement is stopped before crossing a solid cell. A blocked move reverses the heading; the agent is treated as a point for collisions, regardless of its drawn size.

Smooth wall avoidance is controlled by `wall_avoid_radius` and `wall_strength`. `thigmotaxis` adds wall-following along mask walls; it does not yet apply to the outer rectangular boundary. Mask geometry is stretched to fit the arena.

For lower-level work, `Model`, `Environment`, `AgentType`, `Boid` and `metrics` are public. Attach an environment before initialising agents, and pass it to standalone metrics: `metrics.compute(model.groups, model.env)`. A model without an environment uses periodic boundaries. Direct nearest-neighbour metrics without an environment retain the default periodic convention.

## Tutorial and development

[CHIMESim_tutorial.ipynb](CHIMESim_tutorial.ipynb) provides interactive examples and lower-level stepping. Start with the examples above for repeatable headless experiments.

Current work focuses on stabilisation. Later versions will address supported Python versions, a maintained validation suite, research export tools, parameter sweeps and further interaction models. See [CHANGELOG.md](CHANGELOG.md).

## Maintainer and licence

Developed and maintained by Jolle Jolles. Ruiheng Jin contributed implementation work during an internship. Earlier contributions remain recorded in the Git history.

Released under the [MIT licence](LICENSE).
