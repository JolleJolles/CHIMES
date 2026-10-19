from ABSim.agents import AgentType, Boid, BOID, PREY, PREDATOR
from ABSim.model import Model
from ABSim.environment import Environment
from ABSim.simulate import simulate
from ABSim import metrics

__all__ = ["AgentType", "Boid", "BOID", "PREY", "PREDATOR",
           "Model", "Environment", "run", "simulate", "metrics"]


def run(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None):
    """Launch the GUI; Tkinter is only required when this function is called.

    Pass agent types and a seed directly, or load a dictionary/JSON as settings.
    With no settings argument the GUI restores its last session. Explicit
    agent_types, mask and bounded arguments override their saved equivalents.
    log=True returns per-step metrics after closing the window.
    """
    try:
        from ABSim.ui_tk import run as launch
    except ModuleNotFoundError as exc:
        if exc.name not in {"tkinter", "_tkinter"}:
            raise
        raise RuntimeError("The GUI requires Python with Tkinter installed. "
                           "Headless simulate() does not require Tkinter.") from exc
    return launch(agent_types=agent_types, log=log, mask=mask, bounded=bounded,
                  settings=settings, seed=seed)
