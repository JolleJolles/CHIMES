"""
Environment — arena boundary and optional mask-based interior walls.

Performance design
------------------
Bounded arena   : pure analytical formula — O(N), no grid, no precomputation.
Mask walls      : grid stored at approximately four-pixel resolution.
                  Wall distances are computed once on load.
                  Per-agent lookup is O(1).  LOS returns None when no mask is
                  loaded so the model skips the check entirely.

Workflow
--------
  env = Environment(w, h, bounded=True)          # bounded arena, no walls
  env = Environment(w, h, mask="arena.png")      # walls from image
  env.bounded = False                             # toggle live via checkbox
"""

import numpy as np

_SCALE = 4   # canvas pixels per wall-grid cell

from scipy.ndimage import distance_transform_edt as _edt




class Environment:
    """
    Parameters
    ----------
    width, height   : canvas size in pixels
    bounded         : if True, canvas edges act as walls (analytical, O(N))
    mask            : path to a PNG/image — dark pixels become interior walls
    avoid_radius    : distance at which wall repulsion begins (px)
    avoid_strength  : peak repulsion force magnitude at zero distance
    """

    def __init__(self, width, height, bounded=True,
                 mask=None,
                 avoid_radius=50.0,
                 avoid_strength=6.0,
                 thigmotaxis=0.0):
        self.width  = int(width)
        self.height = int(height)
        if self.width < 2 or self.height < 2:
            raise ValueError("Arena width and height must be at least 2 pixels")
        self.bounded        = bounded
        self.avoid_radius   = float(avoid_radius)
        self.avoid_strength = float(avoid_strength)
        self.thigmotaxis    = float(thigmotaxis)

        # Downsampled wall grid — only populated when a mask is loaded
        self._gw   = max(2, self.width  // _SCALE)
        self._gh   = max(2, self.height // _SCALE)
        self._mask_path = None
        self._wall = None   # (gh, gw) bool array
        self._sdf  = None   # (gh, gw) float — distances in canvas-pixel units
        self._dgx  = None   # normalised gradient x (points away from wall)
        self._dgy  = None   # normalised gradient y

        if mask is not None:
            self.load_mask(mask)

    # ------------------------------------------------------------------ #
    # Mask loading
    # ------------------------------------------------------------------ #
    def load_mask(self, path):
        """
        Load a wall mask image.
        White (luminance ≥ 128) = free space.
        Dark  (luminance  < 128) = wall.
        The image is stretched to canvas dimensions.
        Requires Pillow: pip install pillow
        """
        import os
        path = os.path.expanduser(str(path))

        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required for mask images — pip install pillow")

        self._mask_path = path
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
        # Preserve every dark source pixel when reducing the mask resolution.
        self._wall = np.zeros((self._gh, self._gw), dtype=bool)
        y, x = np.nonzero(pixels < 128)
        self._wall[y * self._gh // pixels.shape[0], x * self._gw // pixels.shape[1]] = True
        if self._wall.all():
            raise ValueError("The mask contains no free space at the wall-grid resolution")
        self._compute_sdf()

    def _compute_sdf(self):
        """Compute SDF on the tiny downsampled grid — takes < 1 ms."""
        if self._wall is None:
            return
        spacing = (self.height / self._gh, self.width / self._gw)
        d = (_edt(~self._wall, sampling=spacing) -
             _edt(self._wall, sampling=spacing)).astype(np.float32)
        self._sdf = d
        gy, gx = np.gradient(d)
        mag  = np.hypot(gx, gy)
        safe = np.where(mag > 1e-6, mag, 1.0)
        self._dgx = (gx / safe).astype(np.float32)
        self._dgy = (gy / safe).astype(np.float32)

    @property
    def has_mask(self):
        return self._wall is not None and bool(self._wall.any())

    # ------------------------------------------------------------------ #
    # Physics helpers (called from Model._step_group)
    # ------------------------------------------------------------------ #
    def apply_forces(self, pos, vel, W, H):
        """
        Vectorised wall-avoidance force.  Mutates *vel* in-place.
        pos, vel : (n, 2) float arrays.  W, H : current canvas dimensions.
        """
        ar = self.avoid_radius
        s  = self.avoid_strength
        if ar <= 0 or s <= 0:
            return

        # Analytical boundary — O(N), no grid access
        if self.bounded:
            x, y = pos[:, 0], pos[:, 1]
            vel[:, 0] += np.where(x     < ar, s * (1.0 - x     / ar), 0.0)
            vel[:, 0] -= np.where(W - x < ar, s * (1.0 - (W-x) / ar), 0.0)
            vel[:, 1] += np.where(y     < ar, s * (1.0 - y     / ar), 0.0)
            vel[:, 1] -= np.where(H - y < ar, s * (1.0 - (H-y) / ar), 0.0)

        # Interior mask walls — O(N) SDF lookup on tiny grid
        if self.has_mask and self._sdf is not None:
            xi = np.clip((pos[:, 0] * self._gw / self.width).astype(int), 0, self._gw - 1)
            yi = np.clip((pos[:, 1] * self._gh / self.height).astype(int), 0, self._gh - 1)
            d  = self._sdf[yi, xi]
            near = d < ar
            if near.any():
                nx = self._dgx[yi, xi]
                ny = self._dgy[yi, xi]
                strength = np.where(near, s * (1.0 - d / ar), 0.0)
                vel[:, 0] += nx * strength
                vel[:, 1] += ny * strength

                # Thigmotaxis: add tangential force so agents follow the wall
                if self.thigmotaxis > 0:
                    # tangent = normal rotated 90°; pick sign that aligns with velocity
                    tx, ty = -ny, nx
                    vdot   = vel[:, 0] * tx + vel[:, 1] * ty
                    sign   = np.where(vdot >= 0, 1.0, -1.0)
                    tg     = self.thigmotaxis * strength
                    vel[:, 0] += sign * tx * tg
                    vel[:, 1] += sign * ty * tg

    def _trace(self, start, delta):
        """First wall-hit fraction for each segment, using exact grid traversal."""
        shape = start.shape[:-1]
        start = np.asarray(start, dtype=float).reshape(-1, 2)
        delta = np.asarray(delta, dtype=float).reshape(-1, 2)
        scale = np.array([self._gw / self.width, self._gh / self.height])
        point = start * scale
        direction = delta * scale
        cell = np.floor(point).astype(int)
        step = np.sign(direction).astype(int)
        safe = np.where(direction != 0, direction, 1.)
        interval = np.where(direction != 0, np.abs(1 / safe), np.inf)
        boundary = cell + (step > 0)
        crossing = np.where(direction != 0, (boundary - point) / safe, np.inf)
        fraction = np.ones(len(start))
        active = np.ones(len(start), dtype=bool)

        def wall_at(cells):
            return self._wall[cells[:, 1] % self._gh, cells[:, 0] % self._gw]

        fraction[wall_at(cell)] = 0.
        active &= fraction > 0
        while active.any():
            t = crossing.min(axis=1)
            active &= t <= 1
            if not active.any():
                break
            axes = np.isclose(crossing, t[:, None], rtol=0, atol=1e-12) & active[:, None]
            # At corners check both touching cells, so diagonal motion cannot
            # slip between two solid cells.
            x_cell = cell.copy(); x_cell[:, 0] += step[:, 0]
            y_cell = cell.copy(); y_cell[:, 1] += step[:, 1]
            hit = active & ((axes[:, 0] & wall_at(x_cell)) | (axes[:, 1] & wall_at(y_cell)))
            cell += step * axes
            hit |= active & wall_at(cell)
            fraction[hit] = np.maximum(0., t[hit] - 1e-9)
            active &= ~hit
            crossing = np.where(axes, crossing + interval, crossing)
        return fraction.reshape(shape)

    def los_mask(self, pos_a, pos_b):
        """Check complete sight lines; periodic arenas use shortest wrapped paths."""
        if not self.has_mask:
            return None
        delta = pos_b[None, :, :] - pos_a[:, None, :]
        if not self.bounded:
            size = np.array([self.width, self.height])
            delta -= np.round(delta / size) * size
        start = np.broadcast_to(pos_a[:, None, :], delta.shape)
        return self._trace(start, delta) == 1

    def move(self, pos, vel):
        """Move without crossing walls; reflect heading when movement is blocked."""
        pos = self.project_free(np.asarray(pos, dtype=float).copy())
        displacement = vel.copy()
        fraction = np.ones(len(pos))
        if self.has_mask:
            fraction = self._trace(pos, displacement)
        if self.bounded:
            size = np.array([self.width, self.height])
            for axis in range(2):
                d = displacement[:, axis]
                safe = np.where(d != 0, d, 1.)
                limit = np.where(d > 0, (size[axis] - 1e-8 - pos[:, axis]) / safe,
                                 np.where(d < 0, -pos[:, axis] / safe, np.inf))
                fraction = np.minimum(fraction, np.maximum(0., limit))
        blocked = fraction < 1
        result = pos + displacement * fraction[:, None]
        vel[blocked] *= -1
        if not self.bounded:
            result %= np.array([self.width, self.height])
        return result

    def project_free(self, pos):
        """Move points out of solid cells, for example after a mask resize."""
        if not self.has_mask or not len(pos):
            return pos
        x = np.clip((pos[:, 0] * self._gw / self.width).astype(int), 0, self._gw - 1)
        y = np.clip((pos[:, 1] * self._gh / self.height).astype(int), 0, self._gh - 1)
        inside = self._wall[y, x]
        if inside.any():
            nearest = _edt(self._wall, return_distances=False, return_indices=True)
            yy, xx = nearest[:, y[inside], x[inside]]
            pos[inside, 0] = (xx + .5) * self.width / self._gw
            pos[inside, 1] = (yy + .5) * self.height / self._gh
        return pos

    def sample_positions(self, n, rng):
        """Sample free positions, including for small arenas and complex masks."""
        if self.has_mask:
            cells = np.argwhere(~self._wall)
            chosen = cells[rng.integers(len(cells), size=n)]
            offsets = rng.random((n, 2))
            return np.column_stack(((chosen[:, 1] + offsets[:, 0]) * self.width / self._gw,
                                    (chosen[:, 0] + offsets[:, 1]) * self.height / self._gh))
        x0, x1, y0, y1 = self.spawn_bounds
        return rng.uniform([x0, y0], [x1, y1], size=(n, 2))

    def resize(self, width, height):
        """Rebuild wall geometry when the interactive arena changes size."""
        self.width, self.height = int(width), int(height)
        self._gw = max(2, self.width // _SCALE)
        self._gh = max(2, self.height // _SCALE)
        if self._mask_path is not None:
            self.load_mask(self._mask_path)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def spawn_bounds(self):
        """(x_min, x_max, y_min, y_max) safe spawn region."""
        m = int(self.avoid_radius) + 5 if self.bounded else 50
        mx = min(m, self.width * 0.25)
        my = min(m, self.height * 0.25)
        return mx, self.width - mx, my, self.height - my
