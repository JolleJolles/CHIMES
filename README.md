# Model overview
A 2-D agent-based model, which explores the mechanisms of animal collective behaviour. Boids are spawned by default, while predators can be spawned using a button on the interface. 
The goal for the finished model is to have the following functionalities:
- **Graphics mode and data collection mode** -- It can be run with a simple GUI or run without GUI, and instead collect specified data points over a specified number of simulation runs.
- **Option to model environmental gradients** -- such as the depth of drying streams
- **Option to model individual heterogeneity** -- such as boldness and social-proximity
- **Individual movement behaviours and interaction rules with realism**

<img width="1154" height="787" alt="image" src="https://github.com/user-attachments/assets/8f95c61b-8d95-46fb-83f2-832f2fecaa78" />


# File descriptions
- `agents.py` – class definitions
- `config.py` – global parameters
- `environment.py` – geometric objects, e.g. refuge, feeding patch…
- `model.py` – the core model
- `ui_tk.py` – visualization

- `rules/boid_rules.py` – rules specific to class Boid
- `rules/predator_rules.py` – rules specific to class Predator
- `rules/shared_rules.py` – rules shared by all classes

- `temp/run_baselinemodel.py` – a script that runs the baseline model
- `temp/run_model_tk.py` – example script to run the model (i.e., entry point)
- `temp/boid_predator_v3.py` – baseline model (pre-modularization)


# How to run the model

## Clone and install
Clone the repository<br>

Create a virtual environment (optional but recommended)

Install in editable mode:
```
pip install -e .
```

## How to run the model in visualization mode
Inside the main directory, run:
```
python temp/run_model_tk.py
```
Or run the following in python:
```
from abmsim.ui_tk import run
run()
```

## How to run the model in simulation & data collection mode
To be updated...
