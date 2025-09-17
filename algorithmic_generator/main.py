# algorithmic_generator/main.py

import os
import argparse  # Import the argparse library
import grid
import canvas
import patterns

def main():
    """The main function to generate the Kolam, now driven by command-line arguments."""

    # --- 1. Set up the Argument Parser ---
    parser = argparse.ArgumentParser(description="Generate algorithmic Kolam patterns.")
    
    parser.add_argument('--rows', type=int, default=7, help='Number of rows in the grid.')
    parser.add_argument('--cols', type=int, default=7, help='Number of columns in the grid.')
    parser.add_argument('--pattern', type=str, default='interlocking_loops',
                        choices=['woven_serpentine', 'interlocking_loops'],
                        help='Name of the pattern to generate.')
    
    args = parser.parse_args() # Parse the arguments from the command line

    # --- 2. Use Parsed Arguments for Configuration ---
    # Style Configuration (can still be hardcoded or made into arguments later)
    SPACING = 50
    PADDING = 25
    DOT_SIZE = 10
    LINE_WIDTH = 3
    CURVE_DEPTH = 25
    DOT_COLOR = 'black'
    LINE_COLOR = '#005AC7'

    # File Configuration
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    OUTPUT_FILENAME = f"kolam_{args.pattern}_{args.rows}x{args.cols}.png"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 3. Generate the Kolam using the arguments ---
    width, height = grid.get_grid_dimensions(args.rows, args.cols, SPACING, PADDING)
    image, draw = canvas.create_canvas(width, height)
    dot_coords = grid.generate_dot_coordinates(args.rows, args.cols, SPACING, PADDING)
    canvas.draw_dots(draw, dot_coords, DOT_SIZE, color=DOT_COLOR)

    if args.pattern == "woven_serpentine":
        patterns.draw_woven_serpentine(draw, dot_coords, args.rows, args.cols, SPACING,
                                        LINE_WIDTH, CURVE_DEPTH, LINE_COLOR)
    elif args.pattern == "interlocking_loops":
        patterns.draw_interlocking_loops(draw, dot_coords, args.rows, args.cols, SPACING,
                                          LINE_WIDTH, CURVE_DEPTH, LINE_COLOR)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    image.save(output_path)
    print(f"Kolam with '{args.pattern}' pattern saved to '{output_path}'")


if __name__ == "__main__":
    main()