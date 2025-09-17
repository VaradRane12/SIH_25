# algorithmic_generator/patterns.py

def draw_woven_serpentine(draw, dot_coords, rows, cols, spacing, line_width, curve_depth, color):
    """
    Draws the single, continuous, woven serpentine pattern.
    """
    # Step 1: Generate the serpentine path
    serpentine_path = []
    for r in range(rows):
        row_dots = dot_coords[r*cols : (r+1)*cols]
        if r % 2 == 0:
            serpentine_path.extend(row_dots)
        else:
            serpentine_path.extend(reversed(row_dots))
            
    # Step 2: Draw the path segment by segment using curves
    for i in range(len(serpentine_path) - 1):
        start_point = serpentine_path[i]
        end_point = serpentine_path[i+1]
        
        mid_x = (start_point[0] + end_point[0]) / 2
        mid_y = (start_point[1] + end_point[1]) / 2
        
        if start_point[1] == end_point[1]: # Horizontal segment
            control_point = (mid_x, mid_y - curve_depth)
        else: # Vertical segment
            row_index = start_point[1] // spacing 
            if row_index % 2 == 0:
                control_point = (mid_x + curve_depth, mid_y)
            else:
                control_point = (mid_x - curve_depth, mid_y)

        # To call a function from another file, we need its source
        # For simplicity, we'll pass the 'canvas' module itself
        from canvas import draw_bezier_curve
        draw_bezier_curve(draw, start_point, control_point, end_point, 
                          width=line_width, color=color)
        
# Add this new function to algorithmic_generator/patterns.py

def draw_interlocking_loops(draw, dot_coords, rows, cols, spacing, line_width, curve_depth, color):
    """
    Draws a pattern of interlocking loops around 2x2 groups of dots.
    """
    from canvas import draw_bezier_curve

    # Iterate through the grid, looking at the top-left dot of each 2x2 square
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Get the four dots that form the 2x2 square
            top_left = dot_coords[r * cols + c]
            top_right = dot_coords[r * cols + c + 1]
            bottom_left = dot_coords[(r + 1) * cols + c]
            bottom_right = dot_coords[(r + 1) * cols + c + 1]

            # Define the 4 midpoints that form the diamond shape of the loop
            p_top = ((top_left[0] + top_right[0]) / 2, top_left[1])
            p_bottom = ((bottom_left[0] + bottom_right[0]) / 2, bottom_left[1])
            p_left = (top_left[0], (top_left[1] + bottom_left[1]) / 2)
            p_right = (top_right[0], (top_right[1] + bottom_right[1]) / 2)

            # The dots themselves act as control points to create the curves
            draw_bezier_curve(draw, p_top, top_right, p_right, width=line_width, color=color)
            draw_bezier_curve(draw, p_right, bottom_right, p_bottom, width=line_width, color=color)
            draw_bezier_curve(draw, p_bottom, bottom_left, p_left, width=line_width, color=color)
            draw_bezier_curve(draw, p_left, top_left, p_top, width=line_width, color=color)