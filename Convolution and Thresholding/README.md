# 📸 Manual Image Edge Detection with Python and OpenCV

This project provides a hands-on exploration of image edge detection techniques using Python, NumPy, and OpenCV. Dive deep into the fundamental concepts of:

* **Convolution:** 
    * ⚙️ Implement manual image convolution to apply custom filters, including Gaussian smoothing and Sobel edge detection kernels.
* **Sobel Gradient Calculation:** 
    * 📈 Compute image gradients to identify areas of rapid intensity change, highlighting object edges.
* **Thresholding:** 
    * 🎚️ Apply various thresholding methods:
        * ⚪ Manual single threshold
        * 🔵 Double threshold with hysteresis
        * 🟢 Adaptive thresholding 
    * To isolate edges from the gradient magnitude image.
* **Histogram Analysis:** 
    * 📊 Visualize gradient magnitude histograms to understand the distribution of edge strengths in an image and guide threshold selection.

## ✨ Key Features

* **Clear Code:** 
    * 📝 Well-structured Python code with comments for easy understanding and modification.
* **Educational:** 
    * 🎓 Perfect for learning the fundamentals of image processing and edge detection.
* **Customizable:** 
    * 🛠️ Experiment with different kernel sizes, sigma values, and thresholding techniques to observe their impact on edge detection results.

## 🚀 How to Use

1. Clone the main repository: `git clone https://github.com/unusual9guy/Computer-Vision-Projects`
2. Change directory `cd Computer-Vision-Projects`
3. Choose 'Convolution and Thresholding' `cd Convolution and Thresholding/`
4. Install required libraries: `pip install opencv-python numpy matplotlib`
5. Place your image (e.g., `kitty.bmp`) in the project directory.
6. Run the script: `thresholding.py` 
    * The original image, smoothed image, gradient components, and various thresholded results will be displayed.

## 🔮 Future Enhancements

* **Implement additional edge detection algorithms:** 
    * 🔍 Canny edge detector
* **Explore edge-based object detection or segmentation.**
* **Optimize performance for real-time applications.**

**🤝 Contributions and Adaptations**

Feel free to contribute or adapt this project for your own image processing tasks!

**#ImageProcessing #EdgeDetection #Sobel #Convolution #Thresholding #Python #OpenCV**
