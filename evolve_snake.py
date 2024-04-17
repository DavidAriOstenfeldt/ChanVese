#%% Import libraries
import numpy as np
import skimage.io as io
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
import sklearn.cluster
from patchify import patchify, unpatchify

from probability_curve_evolution import *

#%% Helper functions

def normalize(n):
    l = np.sqrt((n ** 2).sum(axis=1, keepdims = True))
    l[l == 0] = 1
    return n / l


def get_normals(snake):
    """ Returns snake normals. """
    ds = normalize(np.roll(snake, 1, axis=0) - snake) 
    tangent = normalize(np.roll(ds, -1, axis=0) + ds)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return normal 


def remove_intersections(snake, method = 'new'):
    """ Reorder snake points to remove self-intersections.
        Arguments: snake represented by a N-by-2 array.
        Returns: snake.
    """

    N = len(snake)
    closed = snake[np.hstack([np.arange(N), 0])]
    for i in range(N - 2):
        for j in range(i + 2, N):
            if is_crossing(closed[i], closed[i + 1], closed[j], closed[j + 1]):
                # Reverse vertices of smallest loop
                rb, re = (i + 1, j) if j - i < N // 2 else (j + 1, i + N)
                indices = np.arange(rb, re+1) % N                 
                closed[indices] = closed[indices[::-1]]                              
    snake = closed[:-1]
    return snake if is_counterclockwise(snake) else np.flip(snake, axis=0)


def keep_snake_inside(snake, shape):
    """ Contains snake inside the image."""
    snake[:, 0] = np.clip(snake[:, 0], 0, shape[0] - 1)
    snake[:, 1] = np.clip(snake[:, 1], 0, shape[1] - 1)
    return snake

def is_ccw(A, B, C):
    # Check if A, B, C are in counterclockwise order
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def is_crossing(A, B, C, D):
    # Check if line segments AB and CD intersect, not robust but ok for our case
    # https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    return is_ccw(A, C, D) != is_ccw(B, C, D) and is_ccw(A, B, C) != is_ccw(A, B, D)


def is_counterclockwise(snake):
    """ Check if points are ordered counterclockwise."""
    return np.dot(snake[1:, 0] - snake[:-1, 0],
                  snake[1:, 1] + snake[:-1, 1]) < 0

def regularization_matrix(N, alpha, beta):
    """ Matrix for smoothing the snake."""
    s = np.zeros(N)
    s[[-2, -1, 0, 1, 2]] = (alpha * np.array([0, 1, -2, 1, 0]) + 
                    beta * np.array([-1, 4, -6, 4, -1]))
    S = scipy.linalg.circulant(s)  
    return scipy.linalg.inv(np.eye(N) - S)

def create_circle_snake(cx, cy, r, l):
    temp = np.linspace(0, 2 * np.pi, l)
    x = r * np.cos(temp) + cx
    y = r * np.sin(temp) + cy
    
    return np.c_[x, y]

def distribute_points(snake):
    """ Distributes snake points equidistantly."""
    N = len(snake)
    closed = snake[np.hstack([np.arange(N), 0])]
    d = np.sqrt(((np.roll(closed, 1, axis=0) - closed) ** 2).sum(axis=1))
    d = np.cumsum(d)
    d = d / d[-1]  # Normalize to 0-1
    x = np.linspace(0, 1, N, endpoint=False)  # New points
    new =  np.stack([np.interp(x, d, closed[:, i]) for i in range(2)], axis=1) 
    return new



#%% Function for evolving the snake

def evolve_snake(snake, image, B, step_size):
    """ Single step of snake evolution."""
    # Divide image into inner and outer region
    s_mask = polygon2mask(image.shape, snake)
    inner = image[s_mask]
    outer = image[False == s_mask]
    
    # Determine probabilities for Curve Evolution
    P_in, P_out = get_pin_pout(image, snake)
    
    # Determine forces
    N = get_normals(snake)
    deltaP = (P_in[snake.astype(int)[:,0],snake.astype(int)[:,1]]-P_out[snake.astype(int)[:,0],snake.astype(int)[:,1]]) 
    Fext = np.multiply(np.array([deltaP,deltaP]).T,N)
    displacement = step_size * Fext * get_normals(snake)

    snake = snake + displacement  # external part
    snake = B @ snake  # internal part, ordering influenced by 2-by-N representation of snake

    snake = remove_intersections(snake)
    snake = distribute_points(snake)
    snake = keep_snake_inside(snake, image.shape)
    return snake


    

#%%
# Image to be displayed
test_image = 'simple_test.png'
# Load an image
image = io.imread('Data/'+test_image, as_gray=True).astype(np.uint8)


# Settings
N = 250
center = (230,230)
radius = 180
alpha = 0.02
beta = 0.8
step_size = 10000

# Create snakes
snake = create_circle_snake(center[0], center[1], radius, N)
plt.title(test_image)
plt.imshow(image, cmap='gray')
plt.plot(snake[:,0], snake[:,1], c='red')
plt.show()

B = regularization_matrix(N, alpha, beta)

newSnake = evolve_snake(snake, image, B, step_size)

plt.title('New snake')
plt.imshow(image, cmap='gray')
plt.plot(newSnake[:,0], newSnake[:,1], c='red')
plt.show()


# Divide image into inner and outer region
s_mask = polygon2mask(image.shape, newSnake)
inner = image[s_mask]
outer = image[False == s_mask]
    
plt.title('Inner and outer regions')
plt.hist(inner, density=True, bins=255, ls='dashed', lw=3, fc=(1, 0, 0, 0.5))
plt.hist(outer, density=True, bins=255, ls='dashed', lw=3, fc=(0, 0, 1, 0.5))
plt.show()