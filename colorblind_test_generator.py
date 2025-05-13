import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math
import sys

class ColorblindTestGenerator:
    def __init__(self, size=(1280, 1280)):
        """Initialize the generator with given image size."""
        self.size = size
        self.circle_min_size = 1
        self.circle_max_size = 20
    
    def _generate_circle_positions(self):
        """Generate random positions for circles that will form the pattern."""
        positions = []
        width, height = self.size
        current_attempts = 0
        #max_attempts = 12000
        
        #while current_attempts < max_attempts and len(positions) < 12000:
        while len(positions) < 12000:
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(self.circle_min_size, self.circle_max_size)
            
            # Check if the new circle overlaps with existing ones
            overlap = False
            for pos in positions:
                distance = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                if distance < (size + pos[2])/2:
                    overlap = True
                    break
            
            if not overlap:
                positions.append((x, y, size))
            current_attempts += 1
            
        return positions

    def _create_text_mask(self, text, font_size=200):
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

    def generate_test(self, text, test_type="colorblind", font_size=200):
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
        positions = self._generate_circle_positions()
        
        # Create text mask
        text_mask = self._create_text_mask(text, font_size)
        text_mask_array = np.array(text_mask)
        
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
        for x, y, size in positions:
            # Sample the mask at the circle's position
            mask_value = text_mask_array[min(y, self.size[1]-1), min(x, self.size[0]-1)]
            color = color1 if mask_value > 127 else color2
            draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
        
        return image

def main(stem=""):
    # Example usage
    generator = ColorblindTestGenerator()
    
    # Generate a colorblind test
    colorblind_test = generator.generate_test("ABC", "colorblind")
    colorblind_test.save(f"{stem}_colorblind_test.png")
    
    # Generate an anti-colorblind test
    anti_colorblind_test = generator.generate_test("123", "anti-colorblind")
    anti_colorblind_test.save(f"{stem}_anti_colorblind_test.png")

if __name__ == "__main__":
    main( sys.argv[1] if len(sys.argv)> 1 else "" )
