"""Arena drawing geometry and mouse tools, independent of agent behaviour.

Coordinates and stroke widths are fractions of arena size, so drawings resize
with the arena and can be saved directly in JSON settings.
"""
import copy
import numpy as np


def _stroke_points(shape):
    points = np.asarray(shape['points'], dtype=float)
    if len(points) >= 4:
        # Close hand-drawn loops whose ends fall within two brush widths.
        tolerance = max(shape.get('width', .01) * 2, .01)
        if np.linalg.norm(points[-1] - points[0]) <= tolerance:
            points = np.vstack((points, points[0]))
    return points


def rasterise(shapes, width, height, base=None):
    """Rasterise solid freehand strokes, rectangles and ellipses without Pillow."""
    wall = np.zeros((height, width), dtype=bool) if base is None else base.copy()
    yy, xx = np.mgrid[:height, :width]
    x, y = (xx + .5) / width, (yy + .5) / height
    for shape in shapes:
        kind = shape['kind']
        if kind == 'Region':
            for x0, y0, x1, y1 in shape['rectangles']:
                if not all(np.isfinite([x0, y0, x1, y1])) or not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError('Invalid filled region')
                wall[round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = bool(shape['solid'])
            continue
        points = _stroke_points(shape) if kind == 'Brush' else np.asarray(shape['points'], dtype=float)
        stroke = float(shape.get('width', .01))
        if (kind not in {'Brush', 'Rectangle', 'Ellipse'} or points.ndim != 2 or
                points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all() or
                np.any(points < 0) or np.any(points > 1) or not np.isfinite(stroke) or stroke <= 0):
            raise ValueError('Invalid wall drawing')
        if kind == 'Brush':
            # Work in grid coordinates; the minimum radius preserves thin strokes.
            radius = max(.75, stroke * min(width, height) / 2)
            for a, b in zip(points[:-1], points[1:]):
                a, b = a * [width, height], b * [width, height]
                delta = b - a
                t = np.clip(((xx + .5 - a[0]) * delta[0] + (yy + .5 - a[1]) * delta[1]) /
                            max(float(delta @ delta), 1e-20), 0, 1)
                wall |= (xx + .5 - a[0] - t * delta[0]) ** 2 + (yy + .5 - a[1] - t * delta[1]) ** 2 <= radius ** 2
        else:
            lo, hi = points.min(axis=0), points.max(axis=0)
            if kind == 'Rectangle':
                wall |= (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
            else:
                centre = (lo + hi) / 2
                radius = np.maximum((hi - lo) / 2, [.5 / width, .5 / height])
                wall |= ((x - centre[0]) / radius[0]) ** 2 + ((y - centre[1]) / radius[1]) ** 2 <= 1
    return wall


def fill_region(environment, point):
    """Toggle an enclosed grid region; arena edges always stop the flood fill."""
    from scipy.ndimage import label
    wall = environment._wall
    height, width = wall.shape
    x = min(width - 1, max(0, int(point[0] * width)))
    y = min(height - 1, max(0, int(point[1] * height)))
    # Reuse the last painted region when toggling it, preserving its boundary
    # instead of merging the fill with the enclosing black brush strokes.
    for shape in reversed(environment.wall_shapes):
        if shape['kind'] == 'Region' and any(
                x0 <= (x + .5) / width < x1 and y0 <= (y + .5) / height < y1
                for x0, y0, x1, y1 in shape['rectangles']):
            return dict(kind='Region', rectangles=copy.deepcopy(shape['rectangles']),
                        seed=list(point), solid=not bool(wall[y, x]))
    components, _ = label(wall == wall[y, x])
    region = components == components[y, x]
    diff = np.diff(np.pad(region.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rectangles = []
    for row in range(height):
        for left, right in zip(np.where(diff[row] == 1)[0], np.where(diff[row] == -1)[0]):
            rectangles.append([left / width, row / height, right / width, (row + 1) / height])
    return dict(kind='Region', rectangles=rectangles, seed=list(point), solid=not bool(wall[y, x]))


def _filled_image(environment):
    """Repaint fills at visual resolution so coarse cell edges are never exposed."""
    from PIL import Image, ImageDraw
    cached = getattr(environment, '_filled_visual', None)
    if cached is not None:
        return cached
    factor = min(2., 2048 / max(environment.width, environment.height))
    width, height = max(2, round(environment.width * factor)), max(2, round(environment.height * factor))
    image = Image.new('L', (width, height), 255)
    if environment._mask_path is not None:
        with Image.open(environment._mask_path) as source:
            image = source.convert('L').point(lambda value: 0 if value < 128 else 255).resize((width, height))
    footprints = {}
    for shape in environment.wall_shapes:
        painter = ImageDraw.Draw(image)
        if shape['kind'] == 'Region':
            colour = 0 if shape['solid'] else 255
            key = tuple(tuple(rectangle) for rectangle in shape['rectangles'])
            if key not in footprints:
                if 'seed' in shape:
                    x, y = shape['seed']
                    seed = (min(width - 1, max(0, int(x * width))),
                            min(height - 1, max(0, int(y * height))))
                else:
                    # Older saves contain only coarse row rectangles. Use them
                    # to locate the region, never as its displayed boundary.
                    from scipy.ndimage import distance_transform_edt
                    hint = Image.new('L', image.size, 0)
                    hint_painter = ImageDraw.Draw(hint)
                    for x0, y0, x1, y1 in shape['rectangles']:
                        hint_painter.rectangle((round(x0 * width), round(y0 * height),
                                                round(x1 * width) - 1, round(y1 * height) - 1), fill=255)
                    mask = np.asarray(hint) > 0
                    eligible = mask & (np.asarray(image) == (255 if shape['solid'] else 0))
                    if not eligible.any():
                        # A repeated paint can already have its requested colour.
                        footprints[key] = np.zeros((height, width), dtype=bool)
                        continue
                    depth = distance_transform_edt(mask)
                    row, column = np.unravel_index(np.argmax(np.where(eligible, depth, -1)), depth.shape)
                    seed = (int(column), int(row))
                marked = image.copy()
                ImageDraw.floodfill(marked, seed, 128, thresh=0)
                footprints[key] = np.asarray(marked) == 128
            pixels = np.array(image)
            pixels[footprints[key]] = colour
            image = Image.fromarray(pixels)
            continue
        points = _stroke_points(shape) if shape['kind'] == 'Brush' else np.asarray(shape['points'])
        coords = [(float(x * width), float(y * height)) for x, y in points]
        if shape['kind'] == 'Brush':
            stroke = max(1, round(shape.get('width', .01) * min(width, height)))
            painter.line(coords, fill=0, width=stroke, joint='curve')
            radius = stroke / 2
            for x, y in coords:
                painter.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
        else:
            low, high = np.min(coords, axis=0), np.max(coords, axis=0)
            box = (*low, *high)
            if shape['kind'] == 'Ellipse':
                painter.ellipse(box, fill=0)
            else:
                painter.rectangle(box, fill=0)
    environment._filled_visual = image
    return image


def _image_contours(image):
    """Trace display-resolution black regions into cached canvas polygons."""
    solid = np.asarray(image) < 128
    padded = np.pad(solid, 1)
    edges = {}
    def add(y, x, dx0, dy0, dx1, dy1):
        for row, col in zip(y, x):
            a, b = (int(col + dx0), int(row + dy0)), (int(col + dx1), int(row + dy1))
            edges.setdefault(a, set()).add(b)
    add(*np.nonzero(solid & ~padded[:-2, 1:-1]), 0, 0, 1, 0)
    add(*np.nonzero(solid & ~padded[1:-1, 2:]), 1, 0, 1, 1)
    add(*np.nonzero(solid & ~padded[2:, 1:-1]), 1, 1, 0, 1)
    add(*np.nonzero(solid & ~padded[1:-1, :-2]), 0, 1, 0, 0)
    loops = []
    while edges:
        start = next(iter(edges))
        current, previous = start, None
        points = []
        while True:
            points.append(current)
            choices = edges[current]
            if previous is None or len(choices) == 1:
                end = next(iter(choices))
            else:
                dx, dy = current[0] - previous[0], current[1] - previous[1]
                def turn(point):
                    ex, ey = point[0] - current[0], point[1] - current[1]
                    return {(1, 0): 3, (0, 1): 2, (-1, 0): 1, (0, -1): 0}[(dx * ey - dy * ex, dx * ex + dy * ey)]
                end = max(choices, key=turn)
            choices.remove(end)
            if not choices:
                del edges[current]
            previous, current = current, end
            if current == start:
                break
        points = np.asarray(points, dtype=float)
        before, after = points - np.roll(points, 1, axis=0), np.roll(points, -1, axis=0) - points
        points = points[before[:, 0] * after[:, 1] != before[:, 1] * after[:, 0]]
        if len(points) >= 3:
            area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
            loops.append((area, points / [image.width, image.height]))
    return sorted(loops, key=lambda item: -abs(item[0]))


def draw_walls(canvas, environment, transform):
    """Display original wall geometry independently of the coarse collision grid."""
    scale, ox, oy = transform
    width, height = environment.width * scale, environment.height * scale
    canvas._wall_image = None
    filled = any(shape['kind'] == 'Region' for shape in environment.wall_shapes)
    try:
        if filled:
            from PIL import Image
    except ImportError:
        filled = False
    if environment._mask_path is not None or filled:
        from PIL import Image
        if filled:
            alpha = _filled_image(environment).point(lambda value: 255 - value)
        else:
            with Image.open(environment._mask_path) as source:
                alpha = source.convert("L").point(lambda value: 255 if value < 128 else 0)
        contours = getattr(environment, '_display_contours', None)
        if contours is None:
            contours = _image_contours(Image.fromarray(255 - np.asarray(alpha)))
            environment._display_contours = contours
        for area, points in contours:
            coords = [value for x, y in points for value in (ox + x * width, oy + y * height)]
            canvas.create_polygon(*coords, fill='black' if area > 0 else 'white',
                                  outline='', tags='walls')
    for shape in ([] if filled else environment.wall_shapes):
        if shape['kind'] == 'Region':
            for x0, y0, x1, y1 in shape['rectangles']:
                canvas.create_rectangle(ox + x0 * width, oy + y0 * height,
                                        ox + x1 * width, oy + y1 * height,
                                        fill='black' if shape['solid'] else 'white', outline='', tags='walls')
            continue
        points = _stroke_points(shape) if shape['kind'] == 'Brush' else shape['points']
        coords = [value for x, y in points for value in (ox + x * width, oy + y * height)]
        if shape['kind'] == 'Brush':
            canvas.create_line(*coords, fill='black', capstyle='round', joinstyle='round',
                               width=max(1, shape.get('width', .01) * min(width, height)), tags='walls')
        else:
            method = canvas.create_rectangle if shape['kind'] == 'Rectangle' else canvas.create_oval
            method(*coords, fill='black', outline='', tags='walls')


class WallEditor:
    """Explicit edit mode: pause first, preview a gesture, commit on release."""
    def __init__(self, parent, canvas, transform, environment, pause, commit):
        import tkinter as tk
        self.canvas, self.transform, self.environment = canvas, transform, environment
        self.pause, self.commit = pause, commit
        self.points = []
        self.mode = tk.StringVar(value='Off')
        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        tk.Label(row, text='Draw walls:').pack(side=tk.LEFT)
        tk.OptionMenu(row, self.mode, 'Off', 'Brush', 'Rectangle', 'Ellipse', 'Fill', command=self.select).pack(side=tk.LEFT)
        self.width = tk.IntVar(value=12)
        tk.Label(row, text='Width').pack(side=tk.LEFT)
        tk.Spinbox(row, from_=4, to=100, width=4, textvariable=self.width).pack(side=tk.LEFT)
        buttons = tk.Frame(parent)
        buttons.pack(fill=tk.X)
        tk.Button(buttons, text='Undo wall', command=self.undo).pack(side=tk.LEFT)
        tk.Button(buttons, text='Clear drawn walls', command=lambda: self.commit([])).pack(side=tk.LEFT)
        for event, callback in (('<ButtonPress-1>', self.press), ('<B1-Motion>', self.motion),
                                ('<ButtonRelease-1>', self.release)):
            canvas.bind(event, callback, add='+')

    def select(self, value):
        self.points = []
        self.canvas.delete('wall-preview')
        self.canvas.config(cursor='crosshair' if value != 'Off' else '')
        if value != 'Off':
            self.pause()

    def point(self, event):
        scale, ox, oy = self.transform()
        env = self.environment()
        return [float(np.clip((event.x - ox) / scale / env.width, 0, 1)),
                float(np.clip((event.y - oy) / scale / env.height, 0, 1))]

    def press(self, event):
        if self.mode.get() == 'Off':
            return
        self.pause()
        point = self.point(event)
        scale, ox, oy = self.transform()
        env = self.environment()
        if not (ox <= event.x <= ox + env.width * scale and oy <= event.y <= oy + env.height * scale):
            return 'break'
        if self.mode.get() == 'Fill':
            self.commit(copy.deepcopy(env.wall_shapes) + [fill_region(env, point)])
            return 'break'
        self.points = [point]
        return 'break'

    def motion(self, event):
        if not self.points:
            return
        point = self.point(event)
        if self.mode.get() == 'Brush':
            self.points.append(point)
        else:
            self.points = [self.points[0], point]
        self.canvas.delete('wall-preview')
        scale, ox, oy = self.transform()
        env = self.environment()
        coords = [(ox + p[0] * env.width * scale, oy + p[1] * env.height * scale) for p in self.points]
        coords = [v for p in coords for v in p]
        if len(coords) >= 4:
            if self.mode.get() == 'Brush':
                self.canvas.create_line(*coords, fill='black', capstyle='round', joinstyle='round', width=max(1, self.width.get() * scale), tags='wall-preview')
            else:
                method = self.canvas.create_rectangle if self.mode.get() == 'Rectangle' else self.canvas.create_oval
                method(*coords, fill='black', outline='', tags='wall-preview')
        return 'break'

    def release(self, event):
        if not self.points:
            return
        self.motion(event)
        env = self.environment()
        shape = dict(kind=self.mode.get(), points=self.points,
                     width=max(4, self.width.get()) / min(env.width, env.height))
        self.points = []
        self.canvas.delete('wall-preview')
        self.commit(copy.deepcopy(env.wall_shapes) + [shape])
        return 'break'

    def undo(self):
        self.commit(copy.deepcopy(self.environment().wall_shapes[:-1]))
