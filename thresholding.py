import cv2
import numpy as np
import matplotlib.pyplot as plt

    
# Convolving the image
def convolve_image(image, kernel):
    # Getting the  image and kernel dimensions
    image_height = image.shape[0]
    image_width = image.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    

    # Calcultaing padding dimensions for handling edges
    pad_height = (kernel_height - 1) // 2
    pad_width = (kernel_width - 1) // 2

    # Creating padded image
    padded_image = pad_image(image, pad_height, pad_width)  

    # Initializing the output image with zeros
    output_image = np.zeros_like(image)

    # looping reversely is more efficient
    for y in range(image_height):
        for x in range(image_width):
            # Extract the region of interest around the current pixel
            region_of_interest = padded_image[y:y + kernel_height, x:x + kernel_width]
            # Element-wise multiplication and summation 
            output_image[y, x] = np.sum(region_of_interest * kernel)

    return output_image 


def pad_image(image, pad_height, pad_width, pad_value=0):
    padded_height = image.shape[0] + 2 * pad_height
    padded_width = image.shape[1] + 2 * pad_width
    padded_image = np.full((padded_height, padded_width), pad_value, dtype=image.dtype)
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    return padded_image
    

# calculating the x & y gradient and edge-strength 
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
    
    
    # Calculating gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalizing 
    gradient_x = normalize(gradient_x)
    gradient_y = normalize(gradient_y)
    gradient_magnitude = normalize(gradient_magnitude)

    return gradient_x, gradient_y, gradient_magnitude



# Normalization 
def normalize(array):
    min = array.min()
    max = array.max()
    range = max - min
    return (255 * (array - min) / range).astype('uint8')  # Scaling to 0-255, converting to 8-bit 


def gaussian_kernel(sigma, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    center = (kernel_size - 1) // 2  
    x, y = np.mgrid[-center:center+1, -center:center+1]

    # decomposable gaussian 
    g = np.exp((-x**2)/(2*sigma**2)) * np.exp((-y**2)/(2*sigma**2)) 
    return g / g.sum()  # Shorter normalization

# thresholding the image
def threshold_with_one(image, initial_threshold):
    # initializing the image with zeros first
    thresholded_image = np.zeros_like(image)    
    # looping through all the pixel locations 
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > initial_threshold:
                thresholded_image[y, x] = 255
            else :
                thresholded_image[y,x] = 0

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
original_image = cv2.imread('kitty.bmp', cv2.IMREAD_GRAYSCALE)  

if original_image is None:
    print('Could not read image')
    exit(-1)
 
# Creating the normal average kernel
size = 7
normal_avg_kernel = np.ones((size, size)) / (size * size)

 

# creating a weighted average kernel - Gaussian kernel (by choosing sigma)
sigma = 2.5
kernel_size = 7
weighted_avg_kernel = gaussian_kernel(sigma, kernel_size)   


# calculating the blurred image using normal and weighted mean 
smoothed_normal_avg_image = convolve_image(original_image, normal_avg_kernel)
smoothed_weighted_avg_image = convolve_image(original_image, weighted_avg_kernel)


# calculating gradients for different smoothened image 
gradient_x_normal_avg, gradient_y_normal_avg, gradient_magnitude_normal_avg = compute_sobel_gradients(smoothed_normal_avg_image)
gradient_x_weighted_avg, gradient_y_weighted_avg, gradient_magnitude_weighted_avg = compute_sobel_gradients(smoothed_weighted_avg_image)

# Thresholding image 
initial_threshold = 35

# Experiment with different type of smoothing - normal and gaussian
threshold_image_one_normal_avg = threshold_with_one(gradient_magnitude_normal_avg, initial_threshold)
threshold_image_one_weighted_avg = threshold_with_one(gradient_magnitude_weighted_avg, initial_threshold)



# Displaying the images for Normal Average 
images_normal_avg = [original_image, smoothed_normal_avg_image, gradient_x_normal_avg, gradient_y_normal_avg, gradient_magnitude_normal_avg, threshold_image_one_normal_avg]
# Combinig images for normal average into a single big image
combined_image_normal_avg = np.hstack(images_normal_avg)


# Displaying the images for Weighted Average
images_weighted_avg = [original_image, smoothed_weighted_avg_image, gradient_x_weighted_avg, gradient_y_weighted_avg, gradient_magnitude_weighted_avg, threshold_image_one_weighted_avg]
# Combining images for weighted average
combined_image_weighted_avg = np.hstack(images_weighted_avg) 



# Display the combined image
cv2.imshow("Normal Average Images", combined_image_normal_avg)
cv2.imshow("Weighted Average Images", combined_image_weighted_avg)

# plotting histogram
plot_histogram(gradient_magnitude_normal_avg, gradient_magnitude_weighted_avg)

    
while True:
    key = cv2.waitKey(1)  # Capture key press
    if key == ord(' ') or key == 9:  
        break
cv2.destroyAllWindows()


