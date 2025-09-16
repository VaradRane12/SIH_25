import numpy as np
import matplotlib.pyplot as plt
import os

def create_kolam_v2(n=7, img_size=600, sigma=20.0, file_name="kolam_corrected.png"):
    """
    Generates and saves a symmetrical Kolam pattern with corrected dot grid logic.

    Args:
        n (int): The number of dots in the central row (must be an odd number).
        img_size (int): The pixel dimension of the output square image.
        sigma (float): The width of the Gaussian blobs, controlling line spacing.
        file_name (str): The name of the output image file.
    """
    if n % 2 == 0:
        n += 1
        print(f"Input 'n' must be odd. Using n={n} instead.")

    # 1. Create a clear and robust rhombic dot grid with alternating signs.
    dot_positions = []
    dot_signs = []
    
    # The grid is centered at (0,0). Iterate from top row to bottom row.
    # 'i' represents the row index from the center.
    for i in range(-(n // 2), (n // 2) + 1):
        num_dots_in_row = n - 2 * abs(i)
        y = i * 2.0 / n  # Y-coordinate for this row
        
        # 'j' represents the column index within the row.
        for j in range(num_dots_in_row):
            # Calculate the X-coordinate to center the row.
            x = (j * 2.0 - (num_dots_in_row - 1)) / n
            dot_positions.append((x, y))
            
            # Assign sign based on grid position for a checkerboard pattern
            sign = 1 if (i + j) % 2 == 0 else -1
            dot_signs.append(sign)

    # 2. Create the coordinate grid for the image canvas.
    linspace = np.linspace(-1.0, 1.0, img_size)
    X, Y = np.meshgrid(linspace, linspace)
    
    # 3. Calculate the scalar field value at each point.
    Z = np.zeros_like(X)
    for (dot_x, dot_y), sign in zip(dot_positions, dot_signs):
        distance_sq = (X - dot_x)**2 + (Y - dot_y)**2
        # The influence of each dot is a Gaussian function multiplied by its sign.
        Z += sign * np.exp(-distance_sq * (sigma**2))

    # 4. Plot the results.
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Draw the contour line at level 0, which forms the Kolam pattern.
    ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
    
    # Draw the dots on top.
    dots_x, dots_y = zip(*dot_positions)
    ax.scatter(dots_x, dots_y, color='black', s=60, zorder=5)

    # Clean up and save the plot.
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    plt.savefig(file_name, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"✅ Corrected Kolam saved as {file_name}")


if __name__ == '__main__':
    # --- Create a Kolam with a 7-dot center row ---
    # create_kolam_v2(n=7, sigma=25.0, file_name="kolam_7_dot_corrected.png")

    # # --- Create a Kolam with an 11-dot center row ---
    # create_kolam_v2(n=11, sigma=35.0, file_name="kolam_11_dot_corrected.png")
    
    # --- Create a more complex 15-dot Kolam ---
    create_kolam_v2(n=25, sigma=550.0, file_name="kolam_15_dot_corrected.png")