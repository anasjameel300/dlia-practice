# Digital Image Processing Practice Files

This repository contains two comprehensive Python practice files for learning digital image processing using OpenCV and other image processing libraries, plus an ANN classifier experiment on an airline flights dataset.

## Files Included

- **`practice_1_basic_operations.py`** - Basic image operations
- **`practice_2_advanced_operations.py`** - Advanced image processing techniques
- **`requirements.txt`** - Required Python packages
- **`ann_classifier.py`** - 3-layer ANN classifier experiments on `airlines_flights_data.csv`

## Installation

1. Make sure you have Python 3.7 or higher installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Ensure the dataset `airlines_flights_data.csv` is present in the project root (it already is).

## Practice 1: Basic Operations

### Topics Covered:
1. **Reading, Displaying and Saving Image Files**
2. **Resizing Images** - Larger and smaller than original
3. **Flipping Images** - Horizontal, Vertical, and Both
4. **Cropping Images** - 33% of the original image
5. **Color Space Conversion** - Grayscale, BGR, HSV, CMYK, XYZ, LAB
6. **Channel Operations** - Splitting and merging RGB channels
7. **Grayscale Histogram** - Analysis and equalization
8. **Color Histogram** - Analysis and equalization

### Running Practice 1:
```bash
python practice_1_basic_operations.py
```

### Output:
- Creates sample images automatically
- Displays comprehensive visualizations using matplotlib
- Shows all processed images in interactive plots

## Practice 2: Advanced Operations

### Topics Covered:
1. **Arithmetic Operations** - Addition, Subtraction, Blending
2. **Bitwise Operations** - AND, OR, XOR, NOT on binary images
3. **Simple Thresholding** - Binary, Binary Inverse, Truncate, To Zero
4. **Adaptive Thresholding** - Mean and Gaussian methods
5. **Otsu Thresholding** - Automatic threshold selection
6. **Image Segmentation** - Multi-level thresholding
7. **Morphological Operations** - Erosion, Dilation, Opening, Closing
8. **Image Blurring** - Average, Gaussian, Median, Bilateral filtering
9. **Border Creation** - Constant, Reflect, Replicate, Wrap methods
10. **Edge Detection** - Canny, Sobel, Laplacian edge detectors

### Running Practice 2:
```bash
python practice_2_advanced_operations.py
```

### Output:
- Creates sample images automatically
- Displays comprehensive visualizations using matplotlib
- Shows all processed images in interactive plots

## ANN Classifier on Airline Flights Data

Runs binary and multiclass classification using a 3-layer ANN (input → hidden → output) with:

- Hidden units: one configuration < 50 (32) and one > 100 (128)
- Learning rates: 0.0001, 0.001, 0.01
- Reports: accuracy, training time (seconds), and approximate parameter count

### Running the Experiments

```bash
python ann_classifier.py
```

### Outputs

- Console tables for binary and multiclass tasks
- `ann_results.csv` saved in the project root with all experiment results

## Features

- **Automatic Image Generation**: Both scripts create their own sample images, so no external image files are needed
- **Comprehensive Visualization**: All operations are visualized using matplotlib with clear titles and labels
- **Interactive Display**: All processed images are displayed in interactive matplotlib windows
- **Progress Tracking**: Console output shows the progress of each operation
- **No File Dependencies**: Images are created and processed in memory without saving files

## Sample Images Created

Both practice files automatically generate colorful sample images with various shapes and patterns to demonstrate the image processing techniques effectively.

## Learning Objectives

After completing these practices, you will understand:
- Basic image manipulation techniques
- Color space conversions and their applications
- Histogram analysis and equalization
- Various thresholding methods for image segmentation
- Morphological operations for shape analysis
- Different blurring techniques for noise reduction
- Edge detection algorithms for feature extraction
- Bitwise operations for binary image processing

## Notes

- All images are displayed using matplotlib for better visualization (for practice scripts)
- The code includes detailed comments explaining each operation
- Both practice files can be run independently
- No files are saved to disk - everything is displayed in memory
- The ANN script reads `airlines_flights_data.csv` from the repository root

## Troubleshooting

If you encounter any issues:
1. Ensure all required packages are installed correctly
2. Verify that your Python version is 3.7 or higher
3. For display issues, ensure your system supports matplotlib GUI backend
4. If using a remote server or headless environment, you may need to configure the matplotlib backend 