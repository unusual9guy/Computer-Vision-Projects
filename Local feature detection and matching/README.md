# Local Feature Detection and Matching for Object Recognition 🔍

This repository contains a Python implementation for detecting and matching local features in images, a fundamental task in computer vision used for object recognition and image stitching. The code is specifically tailored to identify instances of a particular object (Bernie Sanders) within a set of images. 🖼️

## Key Features 🌟

1. **Harris Corner Detection:** 📐 Implements the Harris corner detection algorithm to identify interest points (corners) in images. These corners are distinctive points that are likely to be found in different views of the same object.

2. **ORB Descriptors:** 🎯 Utilizes ORB (Oriented FAST and Rotated BRIEF) descriptors to create a unique representation of each detected corner. These descriptors capture the appearance of the local image region around the corner, making them suitable for matching across images.

3. **Feature Matching:** 🤝 Includes two methods for matching features between images:
    * **Sum of Squared Differences (SSD):** 📊 A simple but effective method that measures the Euclidean distance between descriptor vectors.
    * **Ratio Test:** ⚖️ A more robust method that compares the distance of the best match to the distance of the second-best match, helping to filter out ambiguous matches.

4. **Visualization:** 📈 Provides functions to visualize the detected keypoints and the matched features between images, aiding in understanding and evaluating the results.

5. **Benchmarking:** 📋 Includes a set of test images with variations (rotation, scale, illumination, blurring) to assess the performance of the feature detector and matcher under different conditions.

## How to Use 🚀

1. Clone the main repository: `git clone https://github.com/unusual9guy/Computer-Vision-Projects`
2. Change directory `cd Computer-Vision-Projects`
3. Choose 'Local-Feature detection and Matching' `cd Local feature detection and matching`
4. **Install Dependencies:** 🛠️
    ```bash
    pip install numpy scipy opencv-python matplotlib
    ```
 5. **Prepare Images:** 🗂️
    * Place your reference image (e.g., `bernieSanders.jpg`) in the `reference images` folder.
    * Place the test images you want to compare in the same folder.

6. **Run the Code:** ▶️
    ```bash
    python main.py
    ```
    * The code will process the images, detect keypoints, compute descriptors, and visualize the matches.
    * Matched images will be saved in the `ratio matches` folder.
    * Plots showing the relationship between the number of keypoints and the threshold value will be saved for each test image.

## Code Structure 🏗️

* **`harris_points_detector(image, sigma=0.5, k=0.04, window_size=7, threshold=0.01)`:** 🎯 Detects Harris corner points in the input image.
* **`non_maximum_suppression(harris_response, window_size, threshold=0.0)`:** 🔍 Refines the detected keypoints by suppressing non-maximal responses.
* **`compute_orb_keypoints(image)`:** 🎯 Computes ORB keypoints for the input image.
* **`compute_orb_descriptors(image, keypoints)`:** 📝 Computes ORB descriptors for the given keypoints.
* **`ssd_feature_matcher(descriptors1, descriptors2)`:** 🤝 Matches features using the SSD distance metric.
* **`ratio_feature_matcher(descriptors1, descriptors2, ratio_threshold=0.75)`:** ⚖️ Matches features using the ratio test.
* **`visualize_matches(image1, keypoints1, image2, keypoints2, matches, save_path=None)`:** 🖼️ Visualizes the matched features between two images.
* **`load_image(image_path)`:** 📥 Loads an image from the specified path.
* **`resize_image(image, max_height=800, max_width=1000)`:** ✂️ Resizes an image if it exceeds the specified dimensions.
* **`get_filename(filepath, type)`:** 📄 Generates filenames for saving results.
* **`calculate_no_of_keypoints(R, orientations, window_size)`:** 🔢 Calculates the number of keypoints based on the Harris response and orientations.
* **`plot_thres_values(images, window_size=7)`:** 📊 Plots the number of keypoints against varying threshold values for each image.

## Key Improvements ✨

* **Clearer Function Names:** 📝 Functions have been renamed to be more descriptive of their purpose.
* **Improved Comments:** 💭 Comments have been added to explain the code's logic and the steps involved in feature detection and matching.
* **Modular Structure:** 🧩 The code is organized into functions for better readability and maintainability.
* **Error Handling:** 🛡️ Basic error handling has been added (e.g., checking if the image exists).

## Future Enhancements 🔮

* **More Advanced Matching Techniques:** 🚀 Explore alternative matching algorithms like FLANN (Fast Library for Approximate Nearest Neighbors).
* **Parameter Tuning:** 🎛️ Experiment with different parameter values (e.g., Harris corner detection parameters, ORB descriptor parameters) to optimize performance.
* **Real-time Implementation:** ⚡ Consider adapting the code for real-time object detection and tracking.

## Disclaimer ⚠️

This code is provided for educational and research purposes. It may not be suitable for production environments without further testing and optimization.
