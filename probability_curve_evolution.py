#%% Import libraries
import numpy as np
import skimage.io as io
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt

#%% Create functions
def create_circle_snake(cx, cy, r, l):
    temp = np.linspace(0, 2 * np.pi, l)
    x = r * np.cos(temp) + cx
    y = r * np.sin(temp) + cy
    
    return np.c_[x, y]

# Following section 2.4 from the paper by Dahl and Dahl
def get_pin_pout(image, snake):

    in_mask = polygon2mask(image.shape, snake).ravel().astype(bool)
    out_mask = 1 - in_mask
    
    # The A matrices are the sum of the pixels in the inner and outer regions
    A_in = np.sum(in_mask)
    A_out = np.sum(out_mask)
    
    # Get the number of pixels in the image
    num_pixels = image.shape[0] * image.shape[1]
    
    # Get the value range (intensities)
    value_range = np.arange(256)
    value_range_matrix = np.tile(value_range, (num_pixels, 1))
    
    # order pixels by column
    image_flat = image.ravel()
    image_matrix = b = np.multiply(image_flat, np.ones((num_pixels, 256), dtype=np.uint8).T).T
    
    # Calculate B matrix
    B = (value_range_matrix == image_matrix).astype(bool)
    
    f_in = (B.T.astype(np.float64) @ in_mask) / A_in
    p_in = f_in / f_in.sum()
    P_in = p_in[image]
    
    f_out = (B.T.astype(np.float64) @ out_mask) / A_out
    p_out = f_out / f_out.sum()
    P_out = p_out[image]
        
    return P_in, P_out


#%% Function to display an image
if __name__ == '__main__':
    # Image to be displayed
    test_image = 'simple_test.png'
    # Load an image
    image = io.imread('Data/'+test_image, as_gray=True).astype(np.uint8)

    # Create snakes
    snake = create_circle_snake(230, 230, 180, 150)
    plt.title(test_image)
    plt.imshow(image, cmap='gray')
    plt.plot(snake[:,0], snake[:,1], c='red')
    plt.show()

    # Divide image into inner and outer region
    s_mask = polygon2mask(image.shape, snake)
    inner = image[s_mask]
    outer = image[False == s_mask]
    
    plt.title('Inner and outer regions')
    plt.hist(inner, density=True, bins=255, ls='dashed', lw=3, fc=(1, 0, 0, 0.5))
    plt.hist(outer, density=True, bins=255, ls='dashed', lw=3, fc=(0, 0, 1, 0.5))
    plt.show()

    # Get pin and pout
    P_in, P_out = get_pin_pout(image, snake)
    
    plt.title('P_in - P_out')
    plt.imshow(P_in - P_out, cmap='RdBu')

#%% 
