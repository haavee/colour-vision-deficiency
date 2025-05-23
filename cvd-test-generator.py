import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math
import sys
import time
import numba
import numpy
import argparse
import re
from enum import Enum

@numba.jit(nopython=True)
def generate_circle_positions(size, min_size, max_size, max_positions, positions):
    """Generate random positions for circles that will form the pattern.
    First fills the image with larger circles, then progressively uses smaller ones.
    """
    width, height = size
    current_attempts = 0
    current_circle = last_check_circle = 0
    # Start with the largest circles and gradually decrease the size
    size_step = math.sqrt(1.4)
    # Try to be smart about the circle size(s)?
    #max_size         = max(max_size, min(width, height)/40)
    #min_size         = min(min(min_size, min(width, height)/100), max_size)
    current_max_size = max_size
    current_min_size = current_max_size / size_step

    while current_circle < max_positions and current_min_size >= min_size:
        # Use the current size range
        size = random.uniform(current_min_size, current_max_size)
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
            current_circle += 1

        current_attempts += 1

        # If we've made many attempts without much progress, reduce the size range
        if current_attempts % 1000 == 0:
            # if, since the last check (1000 attempts ago) 10 or less circles were added, 
            # that's a sign it's getting very difficult to find empty spaces. So we 
            # decrease the circle size and try again
            if ((current_circle - last_check_circle) <= 10):
                current_max_size = current_min_size
                current_min_size = max(min_size, current_min_size / size_step)
                # Reset attempt counter when changing size range
                current_attempts = 0
            last_check_circle = current_circle

    return positions

class TestType(Enum):
    REGULAR_COLORBLIND = "regular_colorblind"  # Standard test - normal vision can see, colorblind struggle
    REVERSE_COLORBLIND = "reverse_colorblind"  # Reverse test - colorblind can see, normal vision struggles
    DEUTERANOPIA = "deuteranopia_test"    # Specific for red-green colorblindness (deuteranopia)
    PROTANOPIA = "protanopia_test"        # Specific for red colorblindness (protanopia)
    TRITANOPIA = "tritanopia_test"        # Specific for blue-yellow colorblindness (tritanopia)
    DEBUG = "debug_test"                  # Just black/white (sort of) to debug

# Color palettes for different types of colorblind tests
ColorPalettes = {

    # Standard Ishihara test colors (normal vision can see, colorblind struggle)
    TestType.REGULAR_COLORBLIND : {
        # Sampled from https://cdn-beaai.nitrocdn.com/DsHNrqyidSdrnEUwxpnDFmLjguAlTfrt/assets/images/optimized/rev-59deba3/colormax.org/wp-content/uploads/2015/08/colorblind-test-image2.jpg
        "background": [(0x6c,0x78,0x3a), (0x7a,0x7c,0x5d), (0xb8,0xbd,0x74), (0xa0,0xa9,0x7a), (0x97,0xa8,0x94), (0x82,0x85,0x4e), (0x72,0x86,0x77)],
        "foreground": [(0xd3,0xa2,0x6e), (0xa0,0x69,0x51), (0xb9,0x87,0x5e), (0xaf,0x8b,0x68), (0xd4,0xa5,0x7f), (0xb6,0x62,0x52)]
    },

    # Reverse test colors (colorblind can see, normal vision struggles)
    TestType.REVERSE_COLORBLIND : {
        # Fig 4. in https://okkl.co.uk/blogs/news/reverse-colorblind-test
        "background": [(0x39,0x6e,0x1f), (0x40,0x67,0x2d), (0x72,0x5d,0x20), (0x6c,0x63,0x20), (0xe8,0x4c,0x0c), (0x3b,0x6b,0x1e), (0x74,0x5e,0x1f)],
        "foreground": [(0xac,0x7b,0x11), (0x5f,0x90,0x1a), (0x6e,0x8e,0x1b), (0x66,0x8b,0x30), (0x49,0x87,0x31), (0x64,0x88,0x18), (0x9c,0x86,0x4e)]
    },

    # Specific for deuteranopia (red-green colorblindness)
    TestType.DEUTERANOPIA : {
        "background": [(0, 128, 128), (128, 128, 0)],  # Teal, Olive - similar for deuteranopes
        "foreground": [(0, 150, 150), (150, 150, 0)]  # Slightly different teal and olive
    },

    # Specific for protanopia (red colorblindness)
    TestType.PROTANOPIA : {
        "background": [(0, 0, 255), (255, 255, 0)],  # Blue, Yellow - high contrast for protanopes
        "foreground": [(50, 50, 255), (255, 255, 50)]  # Slightly different blue and yellow
    },

    # Specific for tritanopia (blue-yellow colorblindness)
    TestType.TRITANOPIA : {
        "background": [(255, 0, 0), (0, 255, 0)],  # Red, Green - high contrast for tritanopes
        "foreground": [(255, 50, 50), (50, 255, 50)]  # Slightly different red and green
    },

    TestType.DEBUG : {
        "background": [(200, 200, 200), (230, 230, 230), (170,170,170) ],  # Nearly white
        "foreground": [(10, 10, 10), (40,40,40), (60,60,60)]  # Nearly black
    }
}


# Add some randomness to colors to create more natural looking patterns
def randomize_color(color, amount=10):
    r, g, b = color
    r = max(0, min(255, r + random.randint(-amount, amount)))
    g = max(0, min(255, g + random.randint(-amount, amount)))
    b = max(0, min(255, b + random.randint(-amount, amount)))
    return (r, g, b)

class ColorblindTestGenerator:
    def __init__(self, size, font, font_size):
        """Initialize the generator with given image size."""
        self.size = size
        self.font_file = font
        self.font_size = font_size
        self.circle_min_size = 2
        self.circle_max_size = 30#25
        self.max_positions = 48000
        self.positions = numpy.zeros((self.max_positions, 3), dtype=numpy.int32)

    def _generate_circle_positions(self):
        print(f"generating circle positions: size={self.size}, min/max size={self.circle_min_size}/{self.circle_max_size}")
        generate_circle_positions(self.size, self.circle_min_size, self.circle_max_size, self.max_positions, self.positions)
        return self.positions

    def _create_text_mask(self, text, font_size=500): #, font_file="Junior-like-IKEA.ttf"):
        """Create a mask from the input text."""
        # Create a new image with a black background
        mask = Image.new('L', self.size, 0)
        draw = ImageDraw.Draw(mask)

        # Try to load a system font
        try:
            font = ImageFont.truetype(self.font_file, self.font_size)
        except:
            # Fallback to default font
            print(f"Failed to load {font_file}, loading default")
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

    def generate_test(self, text, test_type=TestType.REGULAR_COLORBLIND):
        """
        Generate a colorblind test image based on the specified test type.

        Args:
            text (str): The text to display in the test
            test_type (TestType): The type of colorblind test to generate
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
        text_mask = self._create_text_mask(text, self.font_size)
        text_mask_array = np.array(text_mask)
        end = time.time()
        print(f"Mask generation time: {end - start:.2f} seconds")

        # Get color palette based on test type
        if isinstance(t, str):
            # For backward compatibility
            if t == "colorblind":
                t = TestType.REGULAR_COLORBLIND
                #palette = ColorPalettes.REGULAR_COLORBLIND
            elif t == "anti-colorblind":
                #palette = ColorPalettes.REVERSE_COLORBLIND
                t = TestType.REVERSE_COLORBLIND
            else:
                raise ValueError(f"Unknown test type string: {t}")

        palette = ColorPalettes.get(t, None)
        if palette is None:
            raise ValueError(f"Unknown test type: {t}")

        # Get background and foreground colors
        bg_colors = palette["background"]
        fg_colors = palette["foreground"]


        # Draw circles with colors based on the text mask
        for i in range(positions.shape[0]):
            x, y, size = positions[i, :]
            # Sample the mask at the circle's position
            mask_value = text_mask_array[min(y, self.size[1]-1), min(x, self.size[0]-1)]

            if mask_value > 127:
                # Text area - use foreground colors
                base_color = random.choice(fg_colors)
            else:
                # Background area - use background colors
                base_color = random.choice(bg_colors)

            # Add slight randomness to colors for more natural look
            color = base_color #randomize_color(base_color)
            draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)

        return image

    def generate_tests(self, text, test_types):
        """
        Generate colorblind test images based on the specified test type.

        Args:
            text (str): The text to display in the test
            test_types (TestType): list of type of colorblind tests to generate
            font_size (int): Font size for the text

        Returns:
            list of (test_type, PIL.Image): The generated test images
        """
        # Generate circle positions
        start = time.time()
        positions = self._generate_circle_positions()
        end = time.time()
        print(f"Circle generation time: {end - start:.2f} seconds")

        # Create text mask
        start = time.time()
        text_mask = self._create_text_mask(text, self.font_size)
        text_mask_array = np.array(text_mask)
        end = time.time()
        print(f"Mask generation time: {end - start:.2f} seconds")

        images = list()
        for t in test_types:
            # Create the base image
            image = Image.new('RGB', self.size, (255, 255, 255))
            draw = ImageDraw.Draw(image)

            # Get color palette based on test type
            if isinstance(t, str):
                # For backward compatibility
                if t == "colorblind":
                    t = TestType.REGULAR_COLORBLIND
                    #palette = ColorPalettes.REGULAR_COLORBLIND
                elif t == "anti-colorblind":
                    #palette = ColorPalettes.REVERSE_COLORBLIND
                    t = TestType.REVERSE_COLORBLIND
                else:
                    raise ValueError(f"Unknown test type string: {t}")

            palette = ColorPalettes.get(t, None)
            if palette is None:
                raise ValueError(f"Unknown test type: {t}")

            # Get background and foreground colors
            bg_colors = palette["background"]
            fg_colors = palette["foreground"]

            # Draw circles with colors based on the text mask
            for i in range(positions.shape[0]):
                x, y, size = positions[i, :]
                # Sample the mask at the circle's position
                mask_value = text_mask_array[min(y, self.size[1]-1), min(x, self.size[0]-1)]

                if mask_value > 127:
                    # Text area - use foreground colors
                    base_color = random.choice(fg_colors)
                else:
                    # Background area - use background colors
                    base_color = random.choice(bg_colors)

                # Add slight randomness to colors for more natural look
                color = base_color 
                #color = randomize_color(base_color)
                draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
            images.append( (t, image) )

        return images


chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
def rand_string(n):
    return "".join(random.sample(chars, k=n))

def main(text, size, stem, font_file, font_size):
    # Example usage
    generator = ColorblindTestGenerator(size, font_file, font_size)

    # Generate different types of colorblind tests
    #s = "TEST\n\nSTRING"
    for (tp, img) in generator.generate_tests(text or (rand_string(5)+"\n\n"+rand_string(7)),
                                              [TestType.DEBUG, TestType.REGULAR_COLORBLIND, TestType.REVERSE_COLORBLIND]):
        img.save(f"{stem}-{tp}.png")

    ###### Regular colorblind test (normal vision can see, colorblind struggle)
    #regular_test = generator.generate_test("GO\n\nTO\n\nROOM 2.84\n\nFOR\n\nCOOKIES", TestType.REGULAR_COLORBLIND)

    #regular_test = generator.generate_test("TEST\n\nSTRING", TestType.REGULAR_COLORBLIND)
    #regular_test.save(f"{stem}_regular_colorblind_test.png")

    ######### Reverse colorblind test (colorblind can see, normal vision struggles)
    #reverse_test = generator.generate_test("YUMMY\n\nCOOKIES\n\nIN\n\nROOM\n\n6.12", TestType.REVERSE_COLORBLIND)

    #reverse_test = generator.generate_test("TEST\n\nSTRING", TestType.REVERSE_COLORBLIND)
    #reverse_test.save(f"{stem}_reverse_colorblind_test.png")

    # Specific tests for different types of colorblindness
    #deuteranopia_test = generator.generate_test("DEUTERANOPIA\n\nTEST", TestType.DEUTERANOPIA_TEST)
    #deuteranopia_test.save(f"{stem}_deuteranopia_test.png")

    #protanopia_test = generator.generate_test("PROTANOPIA\n\nTEST", TestType.PROTANOPIA_TEST)
    #protanopia_test.save(f"{stem}_protanopia_test.png")

    #tritanopia_test = generator.generate_test("TRITANOPIA\n\nTEST", TestType.TRITANOPIA_TEST)
    #tritanopia_test.save(f"{stem}_tritanopia_test.png")

rxSize  = re.compile(r'^((?P<paper>a4(ls)?)|(?P<w>\d+)x(?P<h>\d+))$', re.I).match
paper_  = { 'a4': (2480, 3840),
            'a4ls': (3840, 2480) }
def mk_size_tuple(s):
    mo = rxSize(s)
    if not mo:
        raise ValueError(f"Size '{s}' not a valid size string")
    paper = mo.group('paper')
    if paper:
        return paper_.get(paper)
    return (int(mo.group('w')), int(mo.group('h')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     description="Colour Vision Deficiency test generator\nCreates filled images with embedded text with random sized dots,using different colour palettes for fore- and background dots, for colour vision deficient (cvd)t test and/or reverse cvd test pattern where a cvd person can read but normal vision fails")
    parser.add_argument('--font-file', dest='font_file', default="Junior-like-IKEA.ttf",
                        help="Point at (true type) font file to use for the text")
    parser.add_argument('--font-size', dest='font_size', default=20, type=int,
                        help="Font size to use for the text")
    parser.add_argument('--size', dest='size', default=(480,480),
                        help="Size of the output image as <width>x<height> pixels. Supports  'a4' for ISO A4 2480x3508",
                        type=mk_size_tuple)
    parser.add_argument('--stem', help=r"Provide basename for output file name, outputs will be named {stem}-{test-type}.png",
                        default="")
    parser.add_argument('text', nargs='*', help="Text to embed")
    args = parser.parse_args()
    main('\n'.join(args.text), args.size, args.stem, args.font_file, args.font_size)
