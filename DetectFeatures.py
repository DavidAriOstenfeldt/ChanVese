# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:40:00 2024

@author: Astrid
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.ndimage

def ind2labels(ind):
    """ Helper function for transforming uint8 image into labeled image."""
    return np.unique(ind, return_inverse=True)[1].reshape(ind.shape)

path = '../ChanVese/Data/' # Change path to your directory

#%%

def get_gauss_feat_im(im, sigma=1, normalize=True):
    """Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r, c).
        sigma: standard deviation for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: 3D array of size (r, c, 15) with a 15-dimentional feature
             vector for every pixel in the image.
    Author: vand@dtu.dk, 2020
    """
      
    r, c = im.shape
    imfeat = np.zeros((r, c, 15))
    imfeat[:, :, 0] = scipy.ndimage.gaussian_filter(im, sigma, order=0)
    imfeat[:, :, 1] = scipy.ndimage.gaussian_filter(im, sigma, order=[0, 1])
    imfeat[:, :, 2] = scipy.ndimage.gaussian_filter(im, sigma, order=[1, 0])
    imfeat[:, :, 3] = scipy.ndimage.gaussian_filter(im, sigma, order=[0, 2])
    imfeat[:, :, 4] = scipy.ndimage.gaussian_filter(im, sigma, order=[1, 1])
    imfeat[:, :, 5] = scipy.ndimage.gaussian_filter(im, sigma, order=[2, 0])
    imfeat[:, :, 6] = scipy.ndimage.gaussian_filter(im, sigma, order=[0, 3])
    imfeat[:, :, 7] = scipy.ndimage.gaussian_filter(im, sigma, order=[1, 2])
    imfeat[:, :, 8] = scipy.ndimage.gaussian_filter(im, sigma, order=[2, 1])
    imfeat[:, :, 9] = scipy.ndimage.gaussian_filter(im, sigma, order=[3, 0])
    imfeat[:, :, 10] = scipy.ndimage.gaussian_filter(im, sigma, order=[0, 4])
    imfeat[:, :, 11] = scipy.ndimage.gaussian_filter(im, sigma, order=[1, 3])
    imfeat[:, :, 12] = scipy.ndimage.gaussian_filter(im, sigma, order=[2, 2])
    imfeat[:, :, 13] = scipy.ndimage.gaussian_filter(im, sigma, order=[3, 1])
    imfeat[:, :, 14] = scipy.ndimage.gaussian_filter(im, sigma, order=[4, 0])

    if normalize:
        imfeat -= np.mean(imfeat, axis=(0, 1))
        im_std = np.std(imfeat, axis=(0, 1))
        im_std[im_std<10e-10] = 1
        imfeat /= im_std
    
    return imfeat

def get_gauss_feat_multi(im, sigma = [1, 2, 4], normalize = True):
    '''Multi-scale Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r, c).
        sigma: list of standard deviations for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: a a 3D array of size (r*c, n_scale, 15) with n_scale features in each pixels, and
             n_scale is length of sigma. Each pixel contains a feature vector and feature
             image is size (r, c, 15*n_scale).
    Author: abda@dtu.dk, 2021

    '''
    imfeats = []
    for i in range(0, len(sigma)):
        feat = get_gauss_feat_im(im, sigma[i], normalize)
        imfeats.append(feat.reshape(-1, feat.shape[2]))
    
    imfeats = np.asarray(imfeats).transpose(1, 0, 2)
    return imfeats


def im2col(im, patch_size=[3, 3], stepsize=1):
    """Rearrange image patches into columns
    Arguments:
        image: a 2D image, shape (r, c).
        patch size: size of extracted paches.
        stepsize: patch step size.
    Returns:
        patches: a 2D array which in every column has a patch associated 
            with one image pixel. For stepsize 1, number of returned column 
            is (r-patch_size[0]+1)*(c-patch_size[0]+1) due to bounary. The 
            length of columns is pathc_size[0]*patch_size[1].
    """
    
    r, c = im.shape
    s0, s1 = im.strides    
    nrows =r - patch_size[0] + 1
    ncols = c - patch_size[1] + 1
    shp = patch_size[0], patch_size[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(im, shape=shp, strides=strd)
    return out_view.reshape(patch_size[0]*patch_size[1], -1)[:, ::stepsize]


def ndim2col(im, block_size=[3, 3], stepsize=1):
    """Rearrange image blocks into columns for N-D image (e.g. RGB image)"""""
    if(im.ndim == 2):
        return im2col(im, block_size, stepsize)
    else:
        r, c, l = im.shape
        patches = np.zeros((l * block_size[0] * block_size[1], 
                            (r - block_size[0] + 1) * (c - block_size[1] + 1)))
        for i in range(l):
            patches[i * block_size[0] * block_size[1] : (i+1) * block_size[0] * block_size[1], 
                    :] = im2col(im[:, :, i], block_size, stepsize)
        return patches

#%% 
# READ IN IMAGES
training_image = skimage.io.imread(path + 'simple_test.png')
training_image = training_image.astype(float)

fig, ax = plt.subplots(1, 1)
ax.imshow(training_image, cmap=plt.cm.gray)
ax.set_title('training image')
fig.tight_layout()
plt.show()

#%% 
# TRAIN THE MODEL

sigma = [1, 2]
features = get_gauss_feat_multi(training_image, sigma)
features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]))


nr_keep = 15000 # number of features randomly picked for clustering 
keep_indices = np.random.permutation(np.arange(features.shape[0]))[:nr_keep]

features_subset = features[keep_indices]

nr_clusters = 1000 # number of feature clusters
# for speed, I use mini-batches
kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=nr_clusters, 
                                         batch_size=2*nr_clusters, 
                                         n_init='auto')
kmeans.fit(features_subset)
assignment = kmeans.labels_

#%%
edges = np.arange(nr_clusters + 1) - 0.5 # histogram edges halfway between integers
hist = np.zeros((nr_clusters, nr_labels))
for l in range(nr_labels):
    hist[:, l] = np.histogram(assignment[labels_subset == l], bins=edges)[0]
sum_hist = np.sum(hist, axis=1)
cluster_probabilities = hist/(sum_hist.reshape(-1, 1))