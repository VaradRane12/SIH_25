import numpy as np
import matplotlib.pyplot as plt
import random # Import the random module

def generate_kolam_field(n, img_size, sign_pattern="checker"):
    """Creates the base scalar field from a rhombic dot grid."""
    if n % 2 == 0:
        n += 1

    dot_positions = []
    dot_signs = []
    
    for i in range(-(n // 2), (n // 2) + 1):
        num_dots_in_row = n - 2 * abs(i)
        y = i * 2.0 / n
        
        for j in range(num_dots_in_row):
            x = (j * 2.0 - (num_dots_in_row - 1)) / n
            dot_positions.append((x, y))
            
            if sign_pattern == "checker":
                sign = 1 if (i + j) % 2 == 0 else -1
            elif sign_pattern == "rows":
                sign = 1 if i % 2 == 0 else -1
            else: 
                sign = 1 if (i + j) % 2 == 0 else -1
            dot_signs.append(sign)

    linspace = np.linspace(-1.2, 1.2, img_size)
    X, Y = np.meshgrid(linspace, linspace)
    
    Z = np.zeros_like(X)
    sigma = (img_size / n) * 1.5
    
    for (dot_x, dot_y), sign in zip(dot_positions, dot_signs):
        distance_sq = (X - dot_x)**2 + (Y - dot_y)**2
        Z += sign * np.exp(-distance_sq * (sigma**2) / (img_size / 100)**2)
        
    return X, Y, Z, dot_positions

def add_layers(Z, num_layers=5, scale_factor=0.5):
    """Adds complexity by layering flipped and scaled versions of the field."""
    if num_layers < 2:
        return Z
        
    base_field = Z.copy()
    layered_field = Z.copy()
    for i in range(1, num_layers):
        flipped_field = base_field[::-1, ::-1]
        layered_field += (scale_factor ** i) * flipped_field
    return layered_field

def draw_kolam(n, img_size, sign_pattern, num_layers, levels, file_name):
    """
    The core drawing function that takes specific design parameters.
    """
    X, Y, Z, dot_positions = generate_kolam_field(n, img_size, sign_pattern)
    Z = add_layers(Z, num_layers=num_layers)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=2)
    
    if dot_positions:
        dots_x, dots_y = zip(*dot_positions)
        ax.scatter(dots_x, dots_y, color='black', s=50, zorder=5)

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(file_name, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"🎨 Random Kolam with n={n} saved as {file_name}")
    print(f"    Parameters used: pattern='{sign_pattern}', layers={num_layers}, levels={levels}")


def draw_random_kolam(n, img_size=800, file_name="kolam_random.png"):
    """
    Generates a Kolam with randomized design parameters.
    """
    # 1. Randomly choose the sign pattern
    random_sign_pattern = random.choice(["checker", "rows"])

    # 2. Randomly choose the number of layers (more chance for simpler designs)
    random_num_layers = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15], k=1)[0]
    
    # 3. Randomly choose the number of contour lines and their spacing
    num_lines = random.choices([1, 3, 5], weights=[0.5, 0.4, 0.1], k=1)[0]
    if num_lines == 1:
        random_levels = [0]
    else:
        spacing = random.uniform(0.08, 0.15)
        if num_lines == 3:
            random_levels = [-spacing, 0, spacing]
        else: # 5 lines
            random_levels = [-2 * spacing, -spacing, 0, spacing, 2 * spacing]

    # Call the main drawing function with the random parameters
    draw_kolam(n=n,
               img_size=img_size,
               sign_pattern=random_sign_pattern,
               num_layers=random_num_layers,
               levels=random_levels,
               file_name=file_name)


if __name__ == '__main__':
    # --- Generate a random Kolam with a 9-dot center row ---
    draw_random_kolam(n=9, file_name="random_kolam_9_dots.png")

    # --- Generate another random one with a 13-dot center row ---
    # Running this multiple times will produce different results
    draw_random_kolam(n=13, file_name="random_kolam_13_dots.png")
    
    # --- Generate a third one to see the variety ---
    draw_random_kolam(n=25, file_name="random_kolam_15_dots.png")