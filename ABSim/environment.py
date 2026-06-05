"""
Environment — arena boundary and optional mask-based interior walls.

Performance design
------------------
Bounded arena   : pure analytical formula — O(N), no grid, no precomputation.
Mask walls      : grid stored at 1/SCALE resolution (SCALE=8 → 240×135 for
                  a 1920×1080 canvas).  SDF computed once on load in <1 ms.
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

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _sdf_numpy(wall):
    """L1 distance transform (4-sweep, pure numpy — no scipy needed).

    Slightly over-estimates Euclidean distance (L1 ≥ L2) but fast and
    sufficient for smooth wall-avoidance forces.  Runs in <1 ms on the
    downsampled grid.
    """
    gh, gw = wall.shape
    INF = float(gh + gw + 2)
    d = np.where(wall, 0.0, INF).astype(np.float32)
    for y in range(1, gh):          d[y]    = np.minimum(d[y],    d[y - 1] + 1)
    for y in range(gh - 2, -1, -1): d[y]    = np.minimum(d[y],    d[y + 1] + 1)
    for x in range(1, gw):          d[:, x] = np.minimum(d[:, x], d[:, x - 1] + 1)
    for x in range(gw - 2, -1, -1): d[:, x] = np.minimum(d[:, x], d[:, x + 1] + 1)
    return d


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
        self.bounded        = bounded
        self.avoid_radius   = float(avoid_radius)
        self.avoid_strength = float(avoid_strength)
        self.thigmotaxis    = float(thigmotaxis)

        # Downsampled wall grid — only populated when a mask is loaded
        self._gw   = max(1, self.width  // _SCALE)
        self._gh   = max(1, self.height // _SCALE)
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

        img_gray = Image.open(path).convert('L')

        # Downsampled binary grid for physics
        small = img_gray.resize((self._gw, self._gh), Image.NEAREST)
        self._wall = np.array(small) < 128

        self._compute_sdf()

    def _compute_sdf(self):
        """Compute SDF on the tiny downsampled grid — takes < 1 ms."""
        if self._wall is None:
            return
        if _HAS_SCIPY:
            d = _edt(~self._wall).astype(np.float32) * _SCALE
        else:
            d = _sdf_numpy(self._wall) * _SCALE
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

        # Analytical boundary — O(N), no grid access
        if self.bounded:
            x, y = pos[:, 0], pos[:, 1]
            vel[:, 0] += np.where(x     < ar, s * (1.0 - x     / ar), 0.0)
            vel[:, 0] -= np.where(W - x < ar, s * (1.0 - (W-x) / ar), 0.0)
            vel[:, 1] += np.where(y     < ar, s * (1.0 - y     / ar), 0.0)
            vel[:, 1] -= np.where(H - y < ar, s * (1.0 - (H-y) / ar), 0.0)

        # Interior mask walls — O(N) SDF lookup on tiny grid
        if self.has_mask and self._sdf is not None:
            xi = np.clip((pos[:, 0] / _SCALE).astype(int), 0, self._gw - 1)
            yi = np.clip((pos[:, 1] / _SCALE).astype(int), 0, self._gh - 1)
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

    def los_mask(self, pos_a, pos_b):
        """
        Midpoint line-of-sight check on the downsampled grid.
        Returns (m, n) bool — True = unobstructed.
        Returns None when no mask is loaded; the model skips the check.
        pos_a : (m, 2),  pos_b : (n, 2)
        """
        if not self.has_mask:
            return None   # no interior walls → skip LOS entirely
        s = 0.5 / _SCALE
        mid_x = ((pos_a[:, None, 0] + pos_b[None, :, 0]) * s).astype(int)
        mid_y = ((pos_a[:, None, 1] + pos_b[None, :, 1]) * s).astype(int)
        np.clip(mid_x, 0, self._gw - 1, out=mid_x)
        np.clip(mid_y, 0, self._gh - 1, out=mid_y)
        return ~self._wall[mid_y, mid_x]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def spawn_bounds(self):
        """(x_min, x_max, y_min, y_max) safe spawn region."""
        m = int(self.avoid_radius) + 5 if self.bounded else 50
        return m, self.width - m, m, self.height - m
