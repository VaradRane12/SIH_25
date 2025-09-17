# algorithmic_generator/canvas.py

from PIL import Image, ImageDraw

def create_canvas(width, height, color='white'):
    """Creates a new blank image canvas."""
    image = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(image)
    return image, draw

def draw_dots(draw, coordinates, dot_size=10, color='black'):
    """Draws dots on a given canvas at specified coordinates."""
    for (x, y) in coordinates:
        dot_bbox = [x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2]
        draw.ellipse(dot_bbox, fill=color)

def draw_path(draw, path_coords, color, width=2):
    """Draws a straight line connecting a sequence of coordinates."""
    if len(path_coords) > 1:
        draw.line(path_coords, fill=color, width=width)

def draw_bezier_curve(draw, start, control, end, width, color, segments=20):
    """Draws a quadratic Bézier curve by approximating it with line segments."""
    curve_points = []
    for i in range(segments + 1):
        t = i / segments
        x = ((1 - t)**2 * start[0]) + (2 * (1 - t) * t * control[0]) + (t**2 * end[0])
        y = ((1 - t)**2 * start[1]) + (2 * (1 - t) * t * control[1]) + (t**2 * end[1])
        curve_points.append((x, y))
    
    draw_path(draw, curve_points, color=color, width=width)