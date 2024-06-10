# Manual Image Edge Detection with Python and OpenCV

This project offers a hands-on exploration of image edge detection techniques using Python, NumPy, and OpenCV. It dives into the fundamental concepts of:

* **Convolution:** Implementation of manual image convolution to apply custom filters, including Gaussian smoothing and Sobel edge detection kernels.
* **Sobel Gradient Calculation:** Computation of image gradients to identify areas of rapid intensity change, highlighting object edges.
* **Thresholding:** Application of various thresholding methods (manual single threshold, double threshold with hysteresis, and adaptive thresholding) to isolate edges from the gradient magnitude image.
* **Histogram Analysis:** Visualization of gradient magnitude histograms to understand the distribution of edge strengths in an image and guide threshold selection.

## Key Features

* **Clear Code:** Well-structured Python code with comments for easy understanding and modification.
* **Educational:** Perfect for learning the fundamentals of image processing and edge detection.
* **Customizable:** Experiment with different kernel sizes, sigma values, and thresholding techniques to observe their impact on edge detection results.

## How to Use

1. Clone the repository.
2. Install required libraries (`opencv-python`, `numpy`, `matplotlib`).
3. Place your image (e.g., `kitty.bmp`) in the project directory.
4. Run the script. The original image, smoothed image, gradient components, and various thresholded results will be displayed.

## Future Enhancements

* Implement additional edge detection algorithms (e.g., Canny).
* Explore edge-based object detection or segmentation.
* Optimize performance for real-time applications.


Feel free to contribute or adapt this project for your own image processing tasks!

**Keywords:** image processing, edge detection, Sobel operator, convolution, thresholding, Python, OpenCV
