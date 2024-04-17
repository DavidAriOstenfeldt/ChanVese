#%%
"""
Representing snake as nx2 array of points.
@author: vand
"""

import numpy as np
import scipy.interpolate
import scipy.linalg
import skimage.draw



def make_circular_snake(N, center, radius):
    """ Initialize circular snake."""
    center = np.asarray(center).reshape([1, 2])
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    unit_circle = np.array([np.cos(angles), np.sin(angles)]).T
    return center + radius * unit_circle


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

    
def regularization_matrix(N, alpha, beta):
    """ Matrix for smoothing the snake."""
    s = np.zeros(N)
    s[[-2, -1, 0, 1, 2]] = (alpha * np.array([0, 1, -2, 1, 0]) + 
                    beta * np.array([-1, 4, -6, 4, -1]))
    S = scipy.linalg.circulant(s)  
    return scipy.linalg.inv(np.eye(N) - S)


def evolve_snake(snake, I, B, step_size):
    """ Single step of snake evolution."""
    mask = skimage.draw.polygon2mask(I.shape, snake)
    m_in = np.mean(I[mask])
    m_out = np.mean(I[~mask])
      
    f = scipy.interpolate.RectBivariateSpline(np.arange(I.shape[0]), np.arange(I.shape[1]), I)
    val = f(snake[:, 0], snake[:, 1], grid=False)

    # val = I[snake[:, 0].astype(int), snake[:, 1].astype(int)]  # simpler variant without interpolation
    f_ext = (m_in - m_out) * (2 * val -m_in - m_out)
    displacement = step_size * f_ext[:,None] * get_normals(snake)

    snake = snake + displacement  # external part
    snake = B @ snake  # internal part, ordering influenced by 2-by-N representation of snake

    snake = remove_intersections(snake)
    snake = distribute_points(snake)
    snake = keep_snake_inside(snake, I.shape)
    return snake


    
