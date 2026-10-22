"""Mouse-wheel bindings for Tk's native platform backends."""


def bind_wheel(widget, callback):
    widget.bind("<MouseWheel>", callback)
    # Button 4/5 wheel events belong to X11; do not register them on Aqua/Win32.
    if widget.tk.call("tk", "windowingsystem") == "x11":
        widget.bind("<Button-4>", callback)
        widget.bind("<Button-5>", callback)


def wheel_units(event):
    number = getattr(event, "num", None)
    if number in (4, 5):
        return -1 if number == 4 else 1
    delta = getattr(event, "delta", 0)
    return -int(delta / 120) if abs(delta) >= 120 else -int(delta)
