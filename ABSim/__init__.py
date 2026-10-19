from ABSim.agents import AgentType, Boid, BOID, PREY, PREDATOR
from ABSim.model import Model
from ABSim.environment import Environment
from ABSim.simulate import simulate
from ABSim import metrics

__all__ = ["AgentType", "Boid", "BOID", "PREY", "PREDATOR",
           "Model", "Environment", "run", "simulate", "metrics"]


def run(*args, **kwargs):
    """Launch the interactive GUI; Tkinter is only required when called."""
    try:
        from ABSim.ui_tk import run as launch
    except ModuleNotFoundError as exc:
        if exc.name not in {"tkinter", "_tkinter"}:
            raise
        raise RuntimeError("The GUI requires Python with Tkinter installed. "
                           "Headless simulate() does not require Tkinter.") from exc
    return launch(*args, **kwargs)
