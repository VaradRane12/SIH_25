# algorithmic_generator/main.py

import os
import grid
import canvas

# --- Configuration ---
ROWS = 7
COLS = 7
SPACING = 50
PADDING = 25
DOT_SIZE = 10
LINE_WIDTH = 3 # New setting for line thickness

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILENAME = "kolam_with_border.png" # Updated filename

def main():
    """The main function to generate the Kolam grid with a border."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    width, height = grid.get_grid_dimensions(ROWS, COLS, SPACING, PADDING)
    image, draw = canvas.create_canvas(width, height)
    dot_coords = grid.generate_dot_coordinates(ROWS, COLS, SPACING, PADDING)
    canvas.draw_dots(draw, dot_coords, DOT_SIZE)
    
    # --- New: Define the Border Path ---
    # Get the coordinates for the four corners
    top_left = dot_coords[0]
    top_right = dot_coords[COLS - 1]
    bottom_right = dot_coords[-1]
    bottom_left = dot_coords[-COLS]
    
    # Create a list of coordinates that form a closed loop
    border_path = [top_left, top_right, bottom_right, bottom_left, top_left]
    
    # --- New: Draw the Path ---
    canvas.draw_path(draw, border_path, width=LINE_WIDTH)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    image.save(output_path)
    print(f"Kolam image with border saved to '{output_path}'")


if __name__ == "__main__":
    main()