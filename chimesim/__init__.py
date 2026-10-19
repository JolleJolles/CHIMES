from chimesim.agents import AgentType, Boid, BOID, PREY, PREDATOR
from chimesim.model import Model
from chimesim.environment import Environment
from chimesim.simulate import simulate
from chimesim import metrics, perception

__all__ = ["AgentType", "Boid", "BOID", "PREY", "PREDATOR",
           "Model", "Environment", "run", "simulate", "metrics", "perception"]


def run(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None,
        width=None, height=None, window_width=None, window_height=None):
    """Launch the GUI; Tkinter is only required when this function is called.

    Pass agent types and a seed directly, or load a dictionary/JSON as settings.
    With no settings argument the GUI restores its last session. Explicit
    agent_types, mask and bounded arguments override their saved equivalents.
    width/height fix the arena size; window_width/window_height size its view.
    Lock arena size prevents view resizing from changing the arena.
    log=True returns per-step metrics; Export recorded data saves recorded trajectories.
    """
    try:
        from chimesim.ui_tk import run as launch
    except ModuleNotFoundError as exc:
        if exc.name not in {"tkinter", "_tkinter"}:
            raise
        raise RuntimeError("The GUI requires Python with Tkinter installed. "
                           "Headless simulate() does not require Tkinter.") from exc
    return launch(agent_types=agent_types, log=log, mask=mask, bounded=bounded,
                  settings=settings, seed=seed, width=width, height=height,
                  window_width=window_width, window_height=window_height)
