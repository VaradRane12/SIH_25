# kolam_generator/grid.py

def generate_dot_coordinates(rows, cols, spacing=50, padding=25):
    """
    Calculates the (x, y) coordinates for a grid of dots.

    Returns:
        list: A list of (x, y) tuples for each dot.
    """
    dot_coordinates = []
    for r in range(rows):
        for c in range(cols):
            x = padding + c * spacing
            y = padding + r * spacing
            dot_coordinates.append((x, y))
    return dot_coordinates

def get_grid_dimensions(rows, cols, spacing=50, padding=25):
    """
    Calculates the total width and height required for the grid image.

    Returns:
        tuple: A tuple containing (width, height).
    """
    width = (cols - 1) * spacing + 2 * padding
    height = (rows - 1) * spacing + 2 * padding
    return (width, height)