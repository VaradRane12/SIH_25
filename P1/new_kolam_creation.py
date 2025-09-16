import numpy as np
import matplotlib.pyplot as plt

def create_kolam(n=7, img_size=512, sigma=150.0, level=0.0, file_name="kolam.png"):
    """
    Generates and saves a Kolam pattern using a scalar field and contour plot.

    Args:
        n (int): The number of dots in the central row (must be an odd number).
        img_size (int): The pixel dimension of the output square image.
        sigma (float): The standard deviation of the Gaussian function, controlling line spacing.
        level (float): The contour level to draw. 0.0 is recommended.
        file_name (str): The name of the output image file.
    """
    if n % 2 == 0:
        print("Warning: n should be an odd number for a symmetrical pattern.")
        n += 1

    # 1. Create the coordinate grid for the image
    x = np.linspace(-1.0, 1.0, img_size)
    y = np.linspace(-1.0, 1.0, img_size)
    X, Y = np.meshgrid(x, y)

    # 2. Generate the rhombic dot locations and their signs
    dot_coords = []
    # Ascending part of the diamond
    for i in range((n // 2) + 1):
        num_dots_in_row = i * 2 + 1
        y_pos = i / (n // 2) - 0.5
        for j in range(num_dots_in_row):
            x_pos = j / (n // 2) - i / (n // 2)
            dot_coords.append((x_pos * 0.8, y_pos * 0.8)) # Scale to fit nicely

    # Mirror for the descending part
    for i in range(len(dot_coords) - n, -1, -n):
         for j in range(i, i+n-2):
            x_pos, y_pos = dot_coords[j]
            if y_pos != 0:
                dot_coords.append((x_pos, -y_pos))
    
    # 3. Calculate the scalar field value at each point in the grid
    Z = np.zeros_like(X)
    for i, (dx, dy) in enumerate(dot_coords):
        # Assign alternating signs like a checkerboard
        sign = 1.0 if (i % (n)) % 2 == 0 else -1.0
        
        # Add the influence of each dot (Gaussian function)
        distance_sq = (X - dx)**2 + (Y - dy)**2
        Z += sign * np.exp(-distance_sq / (2.0 * (sigma / img_size)**2))

    # 4. Plot the results
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the contour lines which form the Kolam pattern
    ax.contour(X, Y, Z, levels=[level], colors='black', linewidths=1.5)
    
    # Draw the dots
    dots_x, dots_y = zip(*dot_coords)
    ax.scatter(dots_x, dots_y, color='black', s=50, zorder=5)

    # Clean up the plot
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.tight_layout()
    
    # Save the file
    plt.savefig(file_name, dpi=300)
    plt.close(fig)
    print(f"✅ Kolam saved as {file_name}")


if __name__ == '__main__':
    # --- Create a Kolam with a 7-dot center row ---
    create_kolam(n=7, sigma=18.0, file_name="kolam_7_dot.png")

    # --- Create a Kolam with a 11-dot center row ---
    create_kolam(n=11, sigma=12.0, file_name="kolam_11_dot.png")