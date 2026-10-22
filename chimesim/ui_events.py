"""Window lifecycle and input helpers for Tk's native platform backends."""


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


def popup_active(widget):
    """Also recognise internal ttk popdowns without Python widget wrappers."""
    return bool(widget.tk.call('grab', 'current', widget._w))


def choice(parent, variable, *values, command=None):
    """Read-only selector; defer mutations until the popup selection has ended."""
    from tkinter import ttk
    widget = ttk.Combobox(parent, textvariable=variable, values=values, state='readonly',
                          width=min(32, max(len(str(v)) for v in values)+1))
    if command is not None:
        def selected(_event):
            value = variable.get()
            def apply():
                if widget.winfo_exists():
                    command(value)
            parent.winfo_toplevel().after_idle(apply)
        widget.bind('<<ComboboxSelected>>', selected)
    return widget


def show_window(window):
    """Map before raising, and bring restored windows back onto the screen."""
    window.deiconify()
    def present():
        if not window.winfo_exists() or window.state()=='withdrawn':
            return
        left, top = window.winfo_vrootx(), window.winfo_vrooty()
        width, height = window.winfo_vrootwidth(), window.winfo_vrootheight()
        x = max(left, min(window.winfo_x(), left+width-min(window.winfo_width(),width)))
        y = max(top+25, min(window.winfo_y(), top+height-min(window.winfo_height(),height)))
        if (x,y)!=(window.winfo_x(),window.winfo_y()):
            window.geometry(f'{x:+d}{y:+d}')
        window.lift()
        window.focus_set()
    window.after_idle(present)


def destroy_window(root):
    """Destroy this Tk session and flush native window closure in notebook kernels."""
    from contextlib import suppress
    from tkinter import TclError
    from _tkinter import DONT_WAIT
    for callback in root.tk.call('after', 'info'):
        with suppress(TclError):
            root.after_cancel(callback)
    try:
        root.destroy()
    finally:
        # Cocoa can leave windows on screen until its queued events are handled.
        # A notebook keeps Python alive without necessarily entering Tk again.
        # Tcl's `update` command may already be unavailable after destruction;
        # use the interpreter event pump directly, without waiting for new input.
        while root.tk.dooneevent(DONT_WAIT):
            pass


def arena_pointer(pointer, transform, arena_size, viewport_size):
    """Convert a pointer inside the viewport and arena to model coordinates."""
    if pointer is None:
        return None
    px, py = pointer
    if not (0 <= px < viewport_size[0] and 0 <= py < viewport_size[1]):
        return None
    scale, ox, oy = transform
    x, y = (px-ox)/scale, (py-oy)/scale
    return (x, y) if 0 <= x < arena_size[0] and 0 <= y < arena_size[1] else None
