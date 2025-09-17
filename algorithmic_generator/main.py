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
LINE_WIDTH = 3
CURVE_DEPTH = 30 # How deep the loops go
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILENAME = "kolam_with_curves.png"

def main():
    """The main function to generate a Kolam with a curved pattern."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    width, height = grid.get_grid_dimensions(ROWS, COLS, SPACING, PADDING)
    image, draw = canvas.create_canvas(width, height)
    dot_coords = grid.generate_dot_coordinates(ROWS, COLS, SPACING, PADDING)
    canvas.draw_dots(draw, dot_coords, DOT_SIZE)
    
    # --- New: Draw a wavy line along the top row ---
    top_row_dots = dot_coords[:COLS] # Get the first row of dots

    for i in range(COLS - 1):
        # Define the start and end points for this segment
        start_point = top_row_dots[i]
        end_point = top_row_dots[i+1]
        
        # The control point will be halfway between them, but offset vertically
        mid_x = (start_point[0] + end_point[0]) / 2
        
        # Alternate between looping up and looping down
        if i % 2 == 0:
            control_point = (mid_x, start_point[1] - CURVE_DEPTH) # Up
        else:
            control_point = (mid_x, start_point[1] + CURVE_DEPTH) # Down
            
        # Tell the artist to draw the curve for this segment
        canvas.draw_bezier_curve(draw, start_point, control_point, end_point, width=LINE_WIDTH)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    image.save(output_path)
    print(f"Kolam image with curves saved to '{output_path}'")


if __name__ == "__main__":
    main()