# Digital Image Processing & Machine Learning Practice Files

This repository contains comprehensive Python practice files for learning digital image processing using OpenCV and machine learning with PyTorch.

## Files Included

- **`practice_1_basic_operations.py`** - Basic image operations
- **`practice_2_advanced_operations.py`** - Advanced image processing techniques
- **`ann_classifier.py`** - 3-Layer Artificial Neural Network for classification
- **`requirements.txt`** - Required Python packages

## Installation

1. Make sure you have Python 3.7 or higher installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

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

## Practice 3: Artificial Neural Network Classification

### Topics Covered:
1. **3-Layer ANN Architecture** - Configurable hidden layers using scikit-learn MLPClassifier
2. **Binary Classification** - Expensive vs Cheap flight prices
3. **Hidden Layer Configurations** - Small (<50 neurons), Large (>100 neurons)
4. **Learning Rate Experiments** - 0.0001, 0.001, 0.01
5. **Performance Metrics** - Accuracy, Confusion Matrix, Training Time
6. **Time Complexity Analysis** - Parameters per second processing
7. **Data Preprocessing** - Feature scaling, categorical encoding
8. **Model Evaluation** - Training/validation/test splits

### Dataset:
- **Airlines Flights Data** - 300,153 flight records with 9 features
- **Classification Task** - Binary classification of flight prices (expensive vs cheap)
- **Features** - Airline, source/destination cities, departure/arrival times, stops, class, duration, days left

### Running Practice 3:
```bash
python ann_classifier.py
```

### Output:
- Comprehensive experiment results with different configurations
- Bar plots showing accuracy and training time comparisons
- Confusion matrices for each configuration
- Summary table comparing all experiments
- Time complexity analysis
- Best configuration identification

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
- **3-Layer ANN architecture using scikit-learn MLPClassifier**
- **Binary classification with neural networks**
- **Impact of hidden layer sizes on model performance**
- **Effect of learning rates on training convergence**
- **Time complexity analysis of neural networks**
- **Data preprocessing for machine learning**
- **Model evaluation and comparison techniques**

## Notes

- All images are displayed using matplotlib for better visualization
- The code includes detailed comments explaining each operation
- Both practice files can be run independently
- No files are saved to disk - everything is displayed in memory

## Troubleshooting

If you encounter any issues:
1. Ensure all required packages are installed correctly
2. Verify that your Python version is 3.7 or higher
3. For display issues, ensure your system supports matplotlib GUI backend
4. If using a remote server or headless environment, you may need to configure the matplotlib backend 