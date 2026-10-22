from chimesim.configuration import settings, settings as _make_settings
from chimesim.simulate import simulate

__all__ = ["settings", "interactive", "simulate"]


def interactive(agent_types=None, log=False, mask=None, bounded=None, settings=None, seed=None,
        width=None, height=None, window_width=None, window_height=None):
    """Launch the GUI; Tkinter is only required when this function is called.

    Pass a scenario from settings(), a dictionary, or a JSON file path.
    With no settings argument the GUI restores its last session. Explicit
    agent_types, mask and bounded arguments override their saved equivalents.
    width/height fix the arena size; window_width/window_height size its view.
    Lock arena size prevents view resizing from changing the arena.
    log=True returns per-step metrics; Export recorded data saves recorded trajectories.
    """
    try:
        from chimesim.ui_tk import interactive as launch
    except ModuleNotFoundError as exc:
        if exc.name not in {"tkinter", "_tkinter"}:
            raise
        raise RuntimeError("The GUI requires Python with Tkinter installed. "
                           "Headless simulate() does not require Tkinter.") from exc
    from chimesim.configuration import flatten
    if settings is not None:
        settings = flatten(settings)
    elif any(v is not None for v in (agent_types, mask, bounded, width, height)):
        settings = flatten(_make_settings(source="defaults"))
    return launch(agent_types=agent_types, log=log, mask=mask, bounded=bounded,
                  settings=settings, seed=seed, width=width, height=height,
                  window_width=window_width, window_height=window_height)
