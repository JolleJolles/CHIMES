"""Recording helpers independent of Tk and simulation movement rules."""
import math


class MouseTrajectory:
    """Sample cursor motion in model coordinates without bridging inactive gaps."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.heading = None

    def sample(self, point, step):
        if point is None:
            self.reset()
            return (0, None, None, None, None, None)
        x, y = point
        dx = dy = None
        if self.previous is not None:
            old_step, old_x, old_y = self.previous
            if step > old_step:
                dx, dy = (x-old_x)/(step-old_step), (y-old_y)/(step-old_step)
                if math.hypot(dx, dy) > 1e-10:
                    self.heading = math.degrees(math.atan2(dy, dx)) % 360
        self.previous = (step, x, y)
        return (1, x, y, self.heading, dx, dy)
