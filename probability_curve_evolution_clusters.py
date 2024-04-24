#%% Import libraries
import numpy as np
import skimage.io as io
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import sklearn.cluster
from patchify import patchify, unpatchify

#%% Create functions
def create_circle_snake(cx, cy, r, l):
    temp = np.linspace(0, 2 * np.pi, l)
    x = r * np.cos(temp) + cx
    y = r * np.sin(temp) + cy
    
    return np.c_[x, y]

# Following section 2.4 from the paper by Dahl and Dahl
def get_pin_pout_cluster(image, snake, M, bins):

    in_mask = polygon2mask(image.shape, snake).ravel().astype(bool)
    out_mask = 1 - in_mask
    
    # The A matrices are the sum of the pixels in the inner and outer regions
    A_in = np.sum(in_mask)
    A_out = np.sum(out_mask)
    
    # Get the number of pixels in the image
    num_pixels = image.shape[0] * image.shape[1]
    
    # Get the value range (intensities)
    value_range = np.arange(bins)
    value_range_matrix = np.tile(value_range, (num_pixels, 1))
    
    # Divide image into patches
    global patches
    patches = patchify(image, (M,M), M)
    patch_size = patches.shape
    patches = patches.reshape(int(image.shape[0]/M)*int(image.shape[1]/M),M*M)
    
    # Cluster patches
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=bins, 
                                         batch_size=2*bins)
    kmeans.fit(patches)
    assignments = kmeans.labels_
    
    # Reshape to image
    global dictionary
    global patch_assignment
    dictionary = kmeans.cluster_centers_
    patch_assignment = assignments.reshape(patch_size[0:2])
    # plt.imshow(patch_assignment)
    # plt.show()
    image_assigment=np.zeros(image.shape)

    for i in range(int(image.shape[0]/M)):
        for j in range(int(image.shape[1]/M)):
            image_assigment[i*M:(i+1)*M,j*M:(j+1)*M]=patch_assignment[i,j]
    image_assigment=image_assigment.astype(np.uint8)

    # plt.imshow(image_assigment)
    # plt.show()
    # Calculate frequencies of bins
    binFrequency = np.histogram(assignments,bins=bins)
    # plt.hist(binFrequency[0],bins=bins)
    # plt.show()
    
    # Calculate dictionary probabilities
    

    # order pixels by column
    image_flat = image_assigment.ravel()
    image_matrix = b = np.multiply(image_flat, np.ones((num_pixels, bins), dtype=np.uint8).T).T
    
    # Calculate B matrix
    
    B = (value_range_matrix == image_matrix).astype(bool)
    

    f_in = (B.T.astype(np.float64) @ in_mask) / A_in
    p_in = f_in / f_in.sum()
    P_in = p_in[image_assigment]
    
    f_out = (B.T.astype(np.float64) @ out_mask) / A_out
    p_out = f_out / f_out.sum()
    P_out = p_out[image_assigment]
        
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
    plt.plot(snake[:,1], snake[:,0], c='red')
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
    P_in, P_out = get_pin_pout_cluster(image, snake, 10, 200)
    
    plt.title('P_in - P_out')
    plt.imshow(P_in - P_out, cmap='RdBu')

#%% 
