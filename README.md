# CHIMESim

**Creating Heterogeneous, Interactive Movement and Environment Simulations.**

A lightweight Python package for exploring collective animal movement. Run a simulation, change parameters with sliders and see the group respond immediately—from coordinated movement to mixed populations with individual differences.

## Explore collective movement

- **Interactive controls:** adjust populations, movement, social responses and environments while simulations run; arrange independent control and information windows to suit your screen.
- **Multiple models:** switch between zonal and Voronoi-neighbour interactions, with optional social acceleration/braking and temporal speed variation.
- **Multiple agent types:** combine up to three types in the GUI, with different behaviours and hunting/fleeing relationships.
- **Individual heterogeneity:** use fixed values or distributions for speed, interaction ranges and other traits; preview distributions and colour agents by their traits.
- **Flexible environments:** choose bounded or periodic arenas, load wall masks or draw obstacles, and adjust wall avoidance and following. Walls block movement and sight.
- **Live information:** follow polarisation, cohesion, speed and spatial groups; explore individual front/back and centre positions over time, ordered by a trait.
- **Data and settings:** record positions and headings, export actual traits and parameter changes, and save/load configurations. Reuse settings for repeated headless simulations.

NumPy/SciPy calculations and filtered wall-visibility checks keep simulations responsive; performance depends on population size, model and environment.

## Get started

From a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e '.[masks]'
```

```python
import chimesim

chimesim.run()
```

Use **Start**, change settings with the sliders, and explore **Individual variation** in each agent-type window. Save a configuration when you find a setup you want to revisit. Window arrangements save automatically.

The GUI requires Python with working Tkinter (`python -m tkinter` checks this). Headless runs do not require Tkinter; install with `python -m pip install -e .` if image masks are not needed. The GUI has been manually tested on macOS Sequoia with Python **3.9, 3.11 and 3.12**; automated model tests also pass on these versions. Python 3.9 remains the declared minimum. Windows/Linux and newer-Python checks run through GitHub Actions; see the [compatibility guide](docs/compatibility.md) for verification status and setup.

## Run repeatable simulations

```python
from chimesim import simulate

results = simulate(n_steps=1000, n_reps=5, burn_in=200, seed=42,
                   save_trajectories=True)
# Or reuse GUI settings: simulate(settings="experiment.json", seed=42)
```

Results contain summaries, optional trajectories and the resolved settings.

## Learn more

- [User guide](docs/usage.md): controls, parameters, heterogeneity, environments, metrics and recording.
- [Tutorial notebook](CHIMESim_tutorial.ipynb): interactive and Python examples. Optional notebook dependencies: `python -m pip install -e '.[notebook]'`.
- [Roadmap and changelog](CHANGELOG.md): planned work and development history.

CHIMESim is under active development. The current zonal and Voronoi models are not exact reproductions of a specific published model; force-based and vision-based models are planned.

Developed and maintained by **Jolle Jolles**, with earlier implementation contributions from Ruiheng Jin during an internship. Released under the [MIT licence](LICENSE).
