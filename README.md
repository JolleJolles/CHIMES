# ABSim

A fast, research-grade 2-D simulation framework for collective animal behaviour, built for behavioural ecologists. ABSim implements the classic Couzin (2002) zone-based flocking model and extends it with wall physics, individual heterogeneity, multiple interaction models, a live interactive GUI, and a headless batch runner for parameter sweeps.

<img width="1154" height="787" alt="ABSim screenshot" src="https://github.com/user-attachments/assets/8f95c61b-8d95-46fb-83f2-832f2fecaa78" />

---

## Why ABSim?

Most general-purpose agent-based modelling tools (such as NetLogo or Mesa) let you build any kind of model, but treat collective movement as a toy example. ABSim is purpose-built for collective behaviour research:

| Feature | ABSim | General tools (e.g. Mesa) |
|---|---|---|
| Vectorised numpy step | ✓ 5–16× faster than loop-based | ✗ Python loops |
| Physical wall / arena system | ✓ smooth force fields | ✗ not available |
| Individual heterogeneity | ✓ sampled distributions, per-agent traits | ✗ |
| Zonal + Voronoi interaction models | ✓ switchable per type | ✗ |
| Live interactive GUI | ✓ native, no browser server | limited |
| Headless batch runner + metrics | ✓ built-in | requires setup |

---

## Key features

### Interaction models
Each agent type can use one of two models for deciding who to respond to:

- **Zonal model** (Couzin 2002): every agent within a given radius counts, split into three zones — a repulsion zone (too close → move away), an orientation zone (medium distance → align headings), and an attraction zone (far → move toward). Classic, well-studied, easy to interpret.
- **Voronoi model**: only *geometric neighbours* count — those with no other agent between you and them. This is a better approximation of how visual perception actually works in a dense group, and is more common in current modelling literature.

### Wall physics
The arena can have an outer boundary and/or interior walls loaded from an image file (dark pixels = wall, light pixels = free space). Agents are pushed away from walls by a smooth repulsion force that scales with proximity — the closer an agent gets, the harder it is pushed back. An optional **thigmotaxis** force adds a tangential component so agents run *along* walls rather than bouncing off them, mimicking wall-following behaviour seen in many animals.

*Behind the scenes:* wall distances are stored in a precomputed grid called a Signed Distance Field (SDF) — essentially a map where every cell stores how far it is from the nearest wall. This means wall forces are an O(1) lookup per agent rather than a geometry calculation, keeping performance high regardless of wall complexity.

Agents can also only perceive neighbours they have an unobstructed line of sight to — a midpoint check on the wall grid blocks perception through walls (Line-of-Sight, LOS check).

### Individual heterogeneity
Any numeric parameter (zone radii, speed, noise, etc.) can be drawn from a statistical distribution across agents rather than being shared. Supported distributions: normal, uniform, lognormal, bimodal. Agents can be coloured by their individual value on a colourmap so you can immediately see, for example, which fish are bold vs shy. Each agent carries a persistent `unique_id` for longitudinal tracking across time steps.

### Interactive GUI
A native Tkinter window — no browser, no server — with:
- Live sliders for all parameters (changes take effect immediately, no reset)
- Resizable sidebar with a draggable divider between canvas and controls
- Steps-per-frame control (×1 ×5 ×20 ×100) to fast-forward without changing the display rate
- Per-agent-type interaction model selector with automatic slider relabelling
- Neighbour connection overlay (toggle to see the Delaunay/Voronoi graph live)
- FPS counter and HUD showing group-level polarisation and nearest-neighbour distance
- Auto-saves all settings to `~/.absim/settings.json` on exit

### Headless batch runner
`simulate()` runs repeated simulations without a GUI, with burn-in, reproducible seeds, and optional full trajectory output — suitable for parameter sweeps and statistical analysis.

---

## Installation

### macOS (recommended path)
The macOS system Python bundles an old version of Tk that renders a blank canvas. Use Homebrew's Python with proper Tk bindings:

```bash
brew install python-tk@3.9
git clone https://github.com/jollejolles/ABSim.git && cd ABSim
/usr/local/bin/python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install jupyter matplotlib   # optional, for the tutorial notebook
```

Verify Tk is working (should print 8.6.x or higher):
```bash
python -c "import tkinter; r=tkinter.Tk(); print(r.tk.call('info','patchlevel')); r.destroy()"
```

### Other platforms
```bash
git clone https://github.com/jollejolles/ABSim.git && cd ABSim
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

**Dependencies** (installed automatically): `numpy`, `scipy`. Optional: `jupyter`, `matplotlib`.

---

## Quick start

### Launch the GUI
```python
from ABSim import run
run()
```

### Run headless and collect data
```python
from ABSim import simulate
import pandas as pd

results = simulate(n_steps=2000, n_reps=10, burn_in=300)
df = pd.DataFrame(results["summary"])
print(df.groupby("rep")["boid_polarisation"].mean())
```

### Custom agent types
```python
from ABSim import BOID, run

shy  = BOID.clone("shy")
bold = BOID.clone("bold")

shy.speed_scale  = 0.8
bold.speed_scale = 1.3
bold.r_r = 5.0   # smaller repulsion zone — tolerates crowding

run(agent_types=[shy, bold])
```

### Individual heterogeneity
```python
from ABSim import BOID, simulate

fish = BOID.clone("fish")
fish.set_distribution("speed",  dist="normal")          # normal around current speed
fish.set_distribution("zoa",    dist="bimodal")         # two social-distance phenotypes
fish.set_distribution("noise",  dist="uniform")         # flat range of noise levels

results = simulate(agent_types=[fish], n_steps=1000, save_trajectories=True)
```

### Voronoi interaction model
```python
from ABSim import BOID, run

fish = BOID.clone("fish")
fish.interaction_model = "voronoi"   # use geometric neighbours instead of metric zones

run(agent_types=[fish])
```

---

## Parameter reference

### AgentType parameters

| Parameter | GUI label | Description |
|---|---|---|
| `r_r` | ZOR / Repulsion threshold | Distance within which an agent moves away from neighbours |
| `r_o` | ZOO / Orientation distance | Distance at which an agent aligns its heading with neighbours |
| `r_a` | ZOA / Attraction radius | Distance at which an agent moves toward neighbours (also max range for Voronoi) |
| `max_turn` | Max turn (°) | Largest heading change allowed per time step — lower = smoother curves |
| `speed_scale` | Speed scale | Multiplier on the global speed slider |
| `social_speed` | Social speed | 0 = fixed speed; >0 = agents match the speed of their orientation neighbours |
| `noise` | Noise σ (°) | Standard deviation of Gaussian heading noise added every step |
| `flee_dist` | Detect range | Distance at which this agent type detects and flees from threats |
| `interaction_model` | Model dropdown | `"zonal"` or `"voronoi"` |

### Global parameters (GUI sliders)

| Slider | Description |
|---|---|
| Speed | Preferred swimming speed for all agents (pixels per step) |
| Field of view (°) | Angular width of each agent's perceptual field (360° = omnidirectional) |
| Catch distance | Predator must be within this distance to catch prey |
| Wall avoid radius | Distance from a wall at which repulsion begins |
| Wall strength | Peak force magnitude of wall repulsion |
| Wall follow | Thigmotaxis strength — how strongly agents follow walls tangentially |

### Environment constructor
```python
Environment(width, height,
            bounded=True,        # canvas edges act as walls
            mask="arena.png",    # image file: dark = wall, light = free
            avoid_radius=50.0,   # wall repulsion onset distance (px)
            avoid_strength=6.0,  # peak repulsion force
            thigmotaxis=0.0)     # wall-following tangential force
```

---

## File structure

```
ABSim/
├── agents.py          Agent type configuration (AgentType) and per-agent state (Boid)
├── model.py           Simulation step: zonal and Voronoi interaction models
├── environment.py     Wall physics: boundary repulsion, mask loading, SDF, LOS
├── simulate.py        Headless batch runner
├── metrics.py         Group-level metrics: polarisation, nearest-neighbour distance
├── ui_tk.py           Interactive GUI (Tkinter)
├── config.py          Global constants (speed limit, FOV, catch distance)
└── functions/
    └── agent_rules.py Cross-group flee, hunt, and predation rules
```

---

## Roadmap

See [CHANGELOG.md](CHANGELOG.md) for the full roadmap. Upcoming additions include a physics/force-based interaction model (Puy et al. 2024 style), retinal-angle neighbour weighting, parameter sweep tools, and a wall mask editor.
