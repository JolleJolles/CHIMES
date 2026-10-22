# CHIMESim

**Creating Heterogeneous, Interactive Movement and Environment Simulations.**

A lightweight Python package for exploring collective animal movement. Run a simulation, change parameters with sliders and see the group respond immediately—from coordinated movement to mixed populations with individual differences.

## Explore collective movement

- **Interactive controls:** adjust populations, movement, social responses and environments while simulations run; arrange independent control and information windows to suit your screen.
- **Multiple models:** switch between zonal and Voronoi-neighbour interactions, with optional social acceleration/braking and temporal speed variation.
- **Multiple agent types:** combine up to three types in the GUI, with different behaviours and hunting/fleeing relationships.
- **Individual heterogeneity:** use fixed values or distributions for speed, interaction ranges and other traits; preview distributions and colour agents by their traits.
- **Flexible environments:** choose bounded or periodic arenas, load wall masks or draw obstacles, and adjust wall avoidance and following. Walls block movement and sight.
- **Starter analysis and plots:** process saved tracks into largest-group metrics and average individual spatial ranks; plot time series, boxplots, densities and trait relationships with optional analysis dependencies.
- **Live information:** follow polarisation, cohesion, speed, milling and spatial groups; explore individual front/back and centre positions over time, ordered by a trait.
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

chimesim.interactive()
```

Use **Start**, change settings with the sliders, and explore **Individual variation** in each agent-type window. Save a configuration when you find a setup you want to revisit. Window arrangements save automatically.

The GUI requires Python with working Tkinter (`python -m tkinter` checks this). Headless runs do not require Tkinter; install with `python -m pip install -e .` if image masks are not needed. The GUI has been manually tested on macOS Sequoia with Python **3.9, 3.11 and 3.12**; automated model tests also pass on these versions. Python 3.9 remains the declared minimum. Windows/Linux and newer-Python checks run through GitHub Actions; see the [compatibility guide](docs/compatibility.md) for verification status and setup.

## Run experiments without Python code

Open the separate setup/progress window from Python:

```python
chimesim.simulate(gui=True)
# Or prefill it: chimesim.simulate(gui=True, settings="experiment.json", n_reps=10)
```

Load saved settings or configure basic agent, arena and movement values, then press **Simulate**. Progress and estimated remaining time are shown; **Stop simulation** preserves written data; **Exit** closes the window. GUI mode saves CSV datasets, settings and status in a new output folder and returns that run information after Exit; pass it to `analysis.load()` or `plotting.plot()`. Direct `simulate(...)` returns datasets to Python. The installed `chimesim-simulate` command opens the same setup window without writing Python code.

## Run repeatable simulations

```python
scenario = chimesim.settings(
    source="defaults", model="zonal",
    environment={"width": 1200, "height": 800, "bounded": True},
    agents={"A1": {"n": 100, "speed_scale": 1.0}},
)
results = chimesim.simulate(settings=scenario, n_steps=1000, n_reps=5, seed=42,
                            save_trajectories=True)
```

The three main functions are **`settings()`**, **`interactive()`** and **`simulate()`**. With no explicit scenario, both interfaces reuse the last interactive scenario; `settings(source="defaults")` gives scripts an independent starting point. Results include per-step and per-repeat summaries, individual trait assignments, optional tracking-style trajectories, the resolved scenario and the actual run seed.

Use `conditions={"agents.A1.speed_scale": [0.5, 1, 1.5]}` for a parameter grid, or configure conditions in `simulate(gui=True)`. Add `output_dir="results"` to stream files into a new `results/run-<timestamp>/` folder beneath the current working directory. Absolute paths are also supported. `burn_in=200` sets unrecorded warm-up steps; raw trajectories are optional and static traits are saved separately.

## Learn more

- [User guide](docs/usage.md): controls, parameters, heterogeneity, environments, metrics and recording.
- [Tutorial notebook](CHIMESim_tutorial.ipynb): interactive and Python examples. Optional notebook dependencies: `python -m pip install -e '.[notebook]'`.
- [Roadmap and changelog](CHANGELOG.md): planned work and development history.

CHIMESim is under active development. The current zonal and Voronoi models are not exact reproductions of a specific published model; force-based and vision-based models are planned.

Developed and maintained by **Jolle Jolles**, with earlier implementation contributions from Ruiheng Jin during an internship. Released under the [MIT licence](LICENSE).
