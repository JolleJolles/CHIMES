"""Asynchronous snapshot visual-network overlay; Tk stays on the main thread."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from chimesim.perception import perceive


def wrapped_segments(start, end, size=None):
    """Split minimum-image connections at arena seams; mark the final segment."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if size is None:
        return [(start, end, True)]
    size = np.asarray(size, dtype=float)
    start %= size
    delta = end - start
    delta -= np.round(delta / size) * size
    stops = [0., 1.]
    for axis in range(2):
        if delta[axis] != 0:
            for boundary in (0., size[axis]):
                t = (boundary - start[axis]) / delta[axis]
                if 0 < t < 1:
                    stops.append(t)
    stops = sorted(set(stops))
    segments = []
    for low, high in zip(stops[:-1], stops[1:]):
        offset = np.floor((start + delta * ((low + high) / 2)) / size) * size
        segments.append((start + delta * low - offset, start + delta * high - offset, high == 1.))
    return segments


class VisualNetwork:
    def __init__(self, canvas, refresh):
        self.canvas, self.refresh = canvas, refresh
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='chimesim-vision')
        self.pending = None
        self.result = None
        self.signature = None
        self.last = -float('inf')
        self.poll = None
        self.closed = False

    def _poll(self):
        self.poll = None
        if self.closed:
            return
        if self.pending is not None and not self.pending.done():
            self.poll = self.canvas.after(50, self._poll)
        else:
            self.refresh()

    def draw(self, agents, environment, transform, enabled, style, fov, radius, threshold, step):
        self.canvas.delete('visual-network')
        if not enabled or self.closed:
            self.result = None
            self.signature = None
            return
        ids = tuple(a.unique_id for a in agents)
        signature = (ids, id(environment), environment.bounded, style, fov, radius)
        if signature != self.signature:
            self.signature = signature
            self.result = None
            self.last = -float('inf')
        if self.pending is not None and self.pending.done():
            result, submitted_signature, submitted_step = self.pending.result()
            self.pending = None
            if submitted_signature == signature:
                self.result = result
                self.result_step = submitted_step
        now = time.perf_counter()
        if self.pending is None and now - self.last >= .5 and (self.result is None or step != self.result_step):
            snapshot = [SimpleNamespace(unique_id=a.unique_id, x=a.x, y=a.y, dx=a.dx, dy=a.dy,
                                        heading=a.heading, type=SimpleNamespace(size=a.type.size)) for a in agents]
            # Snapshot only collision geometry; never copy the cached smooth
            # display image or drawing history into a perception job.
            frozen_env = copy.copy(environment)
            frozen_env._wall = environment._wall.copy()
            frozen_env._filled_visual = None
            frozen_env._display_contours = None
            frozen_env.wall_shapes = []
            def calculate():
                return (perceive(snapshot, frozen_env, body_style=style, fov_deg=fov,
                                 max_distance=radius, samples_per_body=16), signature, step)
            self.pending = self.executor.submit(calculate)
            self.last = now
            if self.poll is None:
                self.poll = self.canvas.after(50, self._poll)
        if self.result is None:
            return
        scale, ox, oy = transform
        by_id = {a.unique_id: a for a in agents}
        visible = self.result.visible & (self.result.visible_fraction >= threshold)
        size = None if environment.bounded else (environment.width, environment.height)
        for i, j in zip(*np.nonzero(visible)):
            source, target = by_id.get(self.result.ids[i]), by_id.get(self.result.ids[j])
            if source is None or target is None:
                continue
            fraction = self.result.visible_fraction[i, j]
            shade = round(180 - 60 * fraction)
            colour = f'#{shade:02x}{shade:02x}{shade:02x}'
            for a, b, final in wrapped_segments((source.x, source.y), (target.x, target.y), size):
                self.canvas.create_line(ox + a[0] * scale, oy + a[1] * scale,
                                        ox + b[0] * scale, oy + b[1] * scale,
                                        fill=colour, stipple='gray50', tags='visual-network')
                if final:
                    tail, tip = a + (b - a) * .45, a + (b - a) * .6
                    self.canvas.create_line(ox + tail[0] * scale, oy + tail[1] * scale,
                                            ox + tip[0] * scale, oy + tip[1] * scale,
                                            fill=colour, stipple='gray50', arrow='last',
                                            arrowshape=(6, 7, 3), tags='visual-network')

    def close(self):
        self.closed = True
        if self.poll is not None:
            self.canvas.after_cancel(self.poll)
        self.executor.shutdown(wait=False, cancel_futures=True)
