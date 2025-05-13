import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math
import sys
import time
import numba
import numpy

@numba.jit(nopython=True)
def generate_circle_positions(size, min_size, max_size, max_positions, positions):
    """Generate random positions for circles that will form the pattern.
    First fills the image with larger circles, then progressively uses smaller ones.
    """
    width, height = size
    current_attempts = 0
    current_circle = 0
    area = 0
    area_target = width * height
    # Start with the largest circles and gradually decrease the size
    size_step = math.sqrt(1.85)
    current_max_size = max_size
    current_min_size = current_max_size / size_step

    while current_circle < max_positions and current_min_size >= min_size:
        # Use the current size range
        size = int(random.uniform(current_min_size, current_max_size))
        x = random.randint(0, width)
        y = random.randint(0, height)

        # Check if the new circle overlaps with existing ones
        overlap = False
        for i in range(current_circle):
            pos = positions[i]
            distance = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if distance < (size + pos[2])/2:
                overlap = True
                break

        if not overlap:
            positions[current_circle, 0] = x
            positions[current_circle, 1] = y
            positions[current_circle, 2] = size
            area += math.pi * size**2
            current_circle += 1

        current_attempts += 1

        # If we've made many attempts without much progress, reduce the size range
        if current_attempts % 1000 == 0:
            if area >= area_target/3:
                current_max_size = current_min_size
                current_min_size = max(min_size, current_min_size / size_step)
                # Reset attempt counter when changing size range
                current_attempts = 0
                area = 0

    return positions

class ColorblindTestGenerator:
    def __init__(self, size=(2480, 3508)):
        """Initialize the generator with given image size."""
        self.size = size
        self.circle_min_size = 2
        self.circle_max_size = 25
        self.max_positions = 48000
        self.positions = numpy.zeros((self.max_positions, 3), dtype=numpy.int32)

    def _generate_circle_positions(self):
        generate_circle_positions(self.size, self.circle_min_size, self.circle_max_size, self.max_positions, self.positions)
        return self.positions

    def _create_text_mask(self, text, font_size=500):
        """Create a mask from the input text."""
        # Create a new image with a black background
        mask = Image.new('L', self.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Try to load a system font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text size and position to center it
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (self.size[0] - text_width) // 2
        y = (self.size[1] - text_height) // 2
        
        # Draw the text in white
        draw.text((x, y), text, fill=255, font=font)
        return mask

    def generate_test(self, text, test_type="colorblind", font_size=400):
        """
        Generate a colorblind or anti-colorblind test image.
        
        Args:
            text (str): The text to display in the test
            test_type (str): Either "colorblind" or "anti-colorblind"
            font_size (int): Font size for the text
        
        Returns:
            PIL.Image: The generated test image
        """
        # Create the base image
        image = Image.new('RGB', self.size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Generate circle positions
        start = time.time()
        positions = self._generate_circle_positions()
        end = time.time()
        print(f"Circle generation time: {end - start:.2f} seconds")

        # Create text mask
        start = time.time()

        text_mask = self._create_text_mask(text, font_size)
        text_mask_array = np.array(text_mask)
        end = time.time()
        print(f"Mask generation time: {end - start:.2f} seconds")

        # Define colors based on test type
        if test_type == "colorblind":
            # Colors that colorblind people will have trouble distinguishing
            color1 = (5, 120, 33)    # Green
            color2 = (215, 25, 28)   # Red
        else:  # anti-colorblind
            # Colors that people with normal vision will have trouble distinguishing
            color1 = (200, 200, 0)   # Yellow
            color2 = (200, 200, 200) # Light gray

        # Draw circles with colors based on the text mask
        for i in range(positions.shape[0]):
            x, y, size = positions[i, :]
            # Sample the mask at the circle's position
            mask_value = text_mask_array[min(y, self.size[1]-1), min(x, self.size[0]-1)]
            color = color1 if mask_value > 127 else color2
            draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)

        return image

def main(stem=""):
    # Example usage
    generator = ColorblindTestGenerator()

    # Generate a colorblind test

    colorblind_test = generator.generate_test("GO\n\nTO\n\nROOM 2.84\n\nFOR\n\nCOOKIES", "colorblind")
    colorblind_test.save(f"{stem}_colorblind_test.png")
    
    # Generate an anti-colorblind test
    anti_colorblind_test = generator.generate_test("YUMMY\n\nCOOKIES\n\nIN\n\nROOM\n\n6.12", "anti-colorblind")
    anti_colorblind_test.save(f"{stem}_anti_colorblind_test.png")

if __name__ == "__main__":
    main( sys.argv[1] if len(sys.argv)> 1 else "" )
