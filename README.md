# BenthicSpeciesDetector
A shape detector for the 2019 MATE underwater robotics competition.

## Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [How to Use](#how-to-use)
  - [Import BenthicSpeciesDetector](#import-benthicspeciesdetector)
  - [Creating a BenthicSpeciesDetector object](#creating-a-benthicspeciesdetector-object)
  - [Modifying attributes](#modifying-attributes)
  - [Detecting benthic species in an image](#detecting-benthic-species-in-an-image)
* [License](#license)

## Overview
BenthicSPeciesDetector is a class that detects squares, rectangles, triangles, and circles/ellipses from an image, for the purpose of completing the benthic species detection task in the 2019 MATE underwater robotics competition.
It accomplishes this through the following sequence of steps:
* Generate object proposals using Selective Search, and filter the proposals to the minimum amount of bounding rectangles that probably contain shapes
* Detect corners within the image, and assign each to the bounding rectangle to which it belongs
* Further filter bounding rectangles which contain too many corners to be a benthic species shape
* Assign a score for each considered benthic species shape to the shape within each bounding rectangle using a combination of corner count, pixel area, and central and interior angles
* Filter shape scores that are below a particular threshold
* Take the highest score for each bounding rectangle as the most probable benthic species

## Features
* Detect the following benthic species (primitives) from a color image for the 2019 MATE underwater robotics competition:
  - Square
  - Rectangle/Bar
  - Triangle
  - Circle/Ellipse

## Requirements
* Python 3.x
* Numpy
* OpenCV 3.x

## Installation
* Install the requirements according to your platform
* Place the benthic_species_detection directory in the same root folder as your project code
  - A wheel version is in progress

## How to Use
### Import BenthicSpeciesDetector
```python
from benthic_species_detection.benthic_species_detector import BenthicSpeciesDetector 
```

### Creating a BenthicSpeciesDetector object
A BenthicSpeciesDetector object can be created by calling the constructor and supplying any combination of the following parameters:
* The ```rect_similarity_alpha``` value; the minimum percent difference for rectangles to be considered dissimilar when filtering
* The ```rect_enclosure_alpha``` value; the minimum percent unenclosed required for a rectangle to be considered unenclosed by another rectangle when filtering
* The ```shape_alpha``` value; the maximum percent difference from ideal shape characteristics when testing considered shapes
* The ```corner_min_distance``` value; the minimum distance (in pixels) two points must be from each other to be considered dissimilar when filtering
```python
detector = BenthicSpeciesDetector(rect_similarity_alpha=0.1, rect_enclosure_alpha=0.1, shape_alpha=0.1, corner_min_distance=10)
```

### Modifying attributes
The four parameters in the constructor can be modified like so:
```python
detector.rect_similarity_alpha = 0.15
detector.rect_enclosure_alpha = 0.20
detector.shape_alpha = 0.15
detector.corner_min_distance = 5
```

### Detecting benthic species in an image
The detector can be used to detect benthic species in an image by calling the ```process``` method and supplying the following parameters:
* image, a numpy matrix in BGR format in which to detect benthic species
* image_scaling_factor, a positive float specifying how the image should be scaled before processing it
* debug, a boolean value which, when enabled, causes the method to print debugging information and draw detection steps in a series of openCV panels
```python
detected_shapes = detector.process(image, image_scaling_factor=0.5, debug=False)
```
The ```process``` method returns a list of all detected shapes in the format ```((x, y, width, height), (shape_type, score))```, where:
* ```(x, y, width, height)``` is the bounding rectangle ((x, y) is the top left corner) of the detected shape scaled back to the original, unscaled input image
* ```shape_type``` is an enum value specified by the detector.ShapeType enum class
* ```score``` is a float value between 0 and 1 representing how likely the considered shape is actually the detected shape type

## License
BenthicSpeciesDetector is distributed under the terms of the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

