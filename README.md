# Colour Vision Deficiency Test Generator

A Python tool for generating custom Ishihara-style colour vision tests with your own text.

## Features

- Generate standard colour vision tests (visible to people with normal vision)
- Create reverse tests (visible to individuals with colour vision deficiency)
- Support for different types of colour vision deficiency:
  - Deuteranopia (red-green colour vision deficiency)
  - Protanopia (red colour vision deficiency)
  - Tritanopia (blue-yellow colour vision deficiency)
- Customize with your own text and font sizes

## Usage

```bash
python cvd-test-generator.py [output_filename_prefix]
```

This will generate multiple test images with different color palettes.

## Requirements

- Python 3.x
- NumPy
- Pillow (PIL)
- Numba
