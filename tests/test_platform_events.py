"""Platform-independent tests of native Tk wheel-event choices."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from chimesim.ui_events import bind_wheel, wheel_units


class PlatformEventTests(unittest.TestCase):
    def test_native_binding_selection(self):
        for backend in ('aqua', 'win32', 'x11'):
            widget = Mock()
            widget.tk.call.return_value = backend
            bind_wheel(widget, lambda _: None)
            events = [call.args[0] for call in widget.bind.call_args_list]
            self.assertEqual(events, ['<MouseWheel>', '<Button-4>', '<Button-5>'] if backend == 'x11' else ['<MouseWheel>'])

    def test_windows_macos_and_x11_wheel_directions(self):
        for event, expected in [(SimpleNamespace(delta=120), -1), (SimpleNamespace(delta=-120), 1),
                                (SimpleNamespace(delta=1), -1), (SimpleNamespace(delta=-1), 1),
                                (SimpleNamespace(num=4), -1), (SimpleNamespace(num=5), 1)]:
            self.assertEqual(wheel_units(event), expected)
