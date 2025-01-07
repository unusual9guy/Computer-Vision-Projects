# Importing the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def displayImage(images, windownames):
    # Calculate the number of rows and columns needed for the subplot grid
    n_images = len(images)
    n_cols = 4  # Changed to 4 columns
    n_rows = 2  # Fixed 2 rows for 8 images

    # Create figure and subplots
    fig = plt.figure(figsize=(20, 10))  # Adjusted figure size for better visibility
    
    # Plot each image
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(windownames[i])
        plt.axis('off')
    
    plt.tight_layout()

    # Add keyboard event handler
    def on_key(event):
        if event.key == ' ' or event.key == 'tab':  # Space or tab
            plt.close('all')
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def convolve_image(image, kernel):
    # Getting the  image and kernel dimensions
    image_height = image.shape[0]
    image_width = image.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # Calcultaing padding for handling edges
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2

    # Creating padded image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initializing the output image with zeros
    output_image = np.zeros_like(image)


    # Looping through each pixel location, the looping is in the opposite order to be more efficient - have to explain properly why 
    for y in range(image_height):
        for x in range(image_width):
            # Extract the region of interest around the current pixel
            region_of_interest = padded_image[y:y + kernel_height, x:x + kernel_width]
            # Element-wise multiplication and summation 
            output_image[y, x] = np.sum(region_of_interest * kernel)

    return output_image 



def compute_sobel_gradients(smoothed_image):

    # Converting to float32 for calculations and to avoid overflow while using the normalization function
    smoothed_image = smoothed_image.astype('float32')

    # Defining Sobel kernels
    sobel_x_kernel = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]])
    
    
    sobel_y_kernel = np.array([[-1, -2, -1], 
                               [ 0,  0,  0], 
                               [ 1,  2,  1]])

    # calculating the gradient_x and gradient_y using the convolution function
    gradient_x = convolve_image(smoothed_image, sobel_x_kernel)
    gradient_y = convolve_image(smoothed_image, sobel_y_kernel)
    
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalizing 
    gradient_x = cv2.normalize(gradient_x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gradient_y = cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return gradient_x, gradient_y, gradient_magnitude


def gaussian_kernel(sigma, kernel_size):
    """Calculates a 2D Gaussian kernel array."""
    center = (kernel_size - 1) / 2
    x, y = np.mgrid[-center:center+1, -center:center+1]

    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / np.sum(g) 

# doing manual thresholding using only one threshold values
def threshold_with_one(image, initial_threshold):
    # initializing the image with zeros first
    thresholded_image = np.zeros_like(image)    
    # looping through all the pixel locations 
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > initial_threshold:
                thresholded_image[y, x] = 255

    return thresholded_image

# doing manual thresholding using two thresholding - using a range
def threshold_with_two(image, low_threshold, high_threshold):
    thresholded_image = np.zeros_like(image)

    def check_neighbors(y, x):
        """Recursive function to explore connected pixels."""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    if thresholded_image[ny, nx] == 255:  # Connected to a strong edge
                        return True
        return False  # Not connected

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > high_threshold:
                thresholded_image[y, x] = 255
            elif image[y, x] >= low_threshold:
                if check_neighbors(y, x):
                    thresholded_image[y, x] = 255

    return thresholded_image

def adaptive_threshold_blur_subtract(image, blur_kernel_size, subtract_weight, threshold):
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    # Subtract a weighted portion of the blurred image
    subtracted_image = image - subtract_weight * blurred_image

    # Apply normal thresholding
    ret, thresholded_image = cv2.threshold(subtracted_image, threshold, 255, cv2.THRESH_BINARY)

    return thresholded_image

def plot_histogram(gradient_magnitude_normal_avg, gradient_magnitude_weighted_avg):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  

    # Plot for normal mean 
    axes[0].hist(gradient_magnitude_normal_avg.ravel(), bins=256, range=[0, 256])
    axes[0].set_title('Gradient Magnitude Histogram - Normal Mean')

    # Plot for weighted mean - gaussian kernel 
    axes[1].hist(gradient_magnitude_weighted_avg.ravel(), bins=256, range=[0, 256], color='green')
    axes[1].set_title('Gradient Magnitude Histogram - Weighted Mean')

    # Adjust spacing and layout for better visibility
    fig.tight_layout()
    plt.show()

# MAIN -------------------------------------------------------------------------------------------------------------------------------

# Loading the image
image = cv2.imread('Images/cat.jpeg', cv2.IMREAD_GRAYSCALE)  

if image is None:
    print('Could not read image')
    exit(-1)
 
# Creating the normal average kernel(no weighted mean)   
avg_kernel = np.ones((3, 3)) /9
# avg_kernel = np.ones((7, 7)) /49
smoothed_image = convolve_image(image, avg_kernel)

# creating a weighted average kernel - Gaussian kernel (by choosing sigma)
sigma = 7 
kernel_size = 3
weighted_avg_kernel = gaussian_kernel(sigma, kernel_size)   



# calculating the blurred/smoothened image
smoothed_weighted_avg_image = convolve_image(image, weighted_avg_kernel)

# computing the gradient_x , gradient_y and gradient magnitude/ edge-strength ------------------------ORIGINAL--------------------------
gradient_x, gradient_y, gradient_magnitude = compute_sobel_gradients(smoothed_image)

# calculating gradients for different smoothened image 
_, _, gradient_magnitude_normal_avg = compute_sobel_gradients(smoothed_image)
_, _, gradient_magnitude_weighted_avg = compute_sobel_gradients(smoothed_weighted_avg_image)

#  Thresholding image 
initial_threshold = 30
# for one threshold value
threshold_image_one = threshold_with_one(gradient_magnitude, initial_threshold)

# Experiment with different type of smoothing
threshold_image_one_normal_avg = threshold_with_one(gradient_magnitude_normal_avg, initial_threshold)
threshold_image_one_weighted_avg = threshold_with_one(gradient_magnitude_weighted_avg, initial_threshold)

cv2.imwrite("Threshold_with_normal_avg.png", threshold_image_one_normal_avg)  
cv2.imwrite("Threshold_with_weighted_avg.png", threshold_image_one_weighted_avg)  

# for two threshold value 
alpha = 64
beta = 120 # have to figure out suitable values
threshold_image_two = threshold_with_two(gradient_magnitude, alpha, beta)

# for adaptive thresholding 
subtract_weight = 0.5 # have to figure out 
kernel_size = 5 # have to figure out 
threshold_adaptive = adaptive_threshold_blur_subtract(gradient_magnitude, kernel_size, subtract_weight, initial_threshold)

# creating a list of images to print
images = [image, smoothed_image, gradient_x, gradient_y, 
          gradient_magnitude, threshold_image_one, threshold_image_two, threshold_adaptive]
# creating a list of window names for the images
windownames = ['Original image', 'Smoothed image', 'Gradient X', 'Gradient Y', 
              'Edge', 'Threshold Image - one', 'Threshold Image - two', 'Adaptive']

# Display all images
displayImage(images, windownames)

# Calculate and plot histogram
plot_histogram(gradient_magnitude_normal_avg, gradient_magnitude_weighted_avg)

    


