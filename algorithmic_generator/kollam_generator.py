from PIL import Image, ImageDraw

def draw_kolam_grid(rows, cols, spacing=50, dot_size=10, padding=25, output_filename="kolam_grid.png"):
    """
    Generates and saves an image of a Kolam dot grid.

    Args:
        rows (int): The number of rows of dots.
        cols (int): The number of columns of dots.
        spacing (int): The distance in pixels between each dot.
        dot_size (int): The diameter of each dot in pixels.
        padding (int): The margin in pixels around the grid.
        output_filename (str): The name of the file to save the image as.
    """
    # Calculate the total width and height needed for the image
    width = (cols - 1) * spacing + 2 * padding
    height = (rows - 1) * spacing + 2 * padding

    # Create a new blank image with a white background
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Loop through each row and column to place the dots
    for r in range(rows):
        for c in range(cols):
            # Calculate the x, y coordinate for the center of each dot
            x = padding + c * spacing
            y = padding + r * spacing

            # Define the top-left and bottom-right corners of the dot's bounding box
            dot_bbox = [x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2]
            
            # Draw the dot as a filled black circle (ellipse)
            draw.ellipse(dot_bbox, fill='black')

    # Save the final image
    image.save(output_filename)
    print(f"Grid image saved as '{output_filename}'")


# This part of the script runs when you execute the file directly
if __name__ == "__main__":
    # Let's create a 7x7 grid to start
    draw_kolam_grid(rows=7, cols=7)