import cv2
import numpy as np

from matplotlib import pyplot as plt
from warnings import warn

def filter2D(input_arr, filter):
    """
    2D filtering (i.e. convolution but without mirroring the filter).  Mostly a convenience wrapper
    around OpenCV.

    Parameters
    ----------
    input_arr : numpy array, HxW size
    filter : numpy array, H1xW1 size
    
    Returns
    -------
    result : numpy array, HxW size

    """
    return cv2.filter2D(input_arr, 
                        -1, 
                        filter,
                        borderType=cv2.BORDER_CONSTANT)

def batch_filter3D(input_arr, filters):
    """
    3D filtering (i.e. convolution but without mirroring the filter).

    Parameters
    ----------
    input_arr : numpy array, NxHxWxC size where N is the number of images to be filtered
    filter : numpy array, H1xW1xC size
    
    Returns
    -------
    result : numpy array, NxHxW size
    
    """
    assert input_arr.shape[3] == filters.shape[2]
    num_input = input_arr.shape[0]
    output = np.zeros(input_arr.shape[:3] + (filters.shape[-1],))
    for n in xrange(num_input):
        input1 = input_arr[n]
        for f in xrange(filters.shape[-1]):
            for c in xrange(filters.shape[-2]):
                output[n,:,:,f] += filter2D(input1[...,c].copy(), filters[...,c,f].copy())
    return output

def padarray(arr, amount):
    """
    Pad array by some amount in height and width dimensions.

    Parameters
    ----------
    arr : numpy array, HxW size
    amount : (int, int) tuple specifying padding amounts in height and width dimensions.  Padding
        be added on all 4 sides.
    
    Returns
    -------
    result : numpy array, (H+2*H_pad)x(W+2*W_pad)

    """
    padded = np.zeros(arr.shape[0:1] +
                      (arr.shape[1]+2*amount[0], arr.shape[2]+2*amount[1]) +
                      arr.shape[3:])
    padded[:, amount[0]:-amount[0], amount[1]:-amount[1], ...] = arr
    return padded

def atleast(arr, ndim=4):
    """
    Increase number of dimensions by adding singleton dimensions.

    Parameters
    ----------
    arr : numpy array
    ndim : desired number of dimensions

    Returns
    -------
    result : numpy array where result.ndim == ndim, size 1 x 1 x 1 ... arr.shape (number of leading
        ones will depend on ndim).

    """
    while arr.ndim < ndim:
        arr = arr[np.newaxis,...]
    return arr

def safe_exp(v):
    """
    Safely apply np.exp.  If the input is beyond a "safe" range, it will be clipped and a warning
    will be issued.

    """
    v = np.array(v)
    if np.any(v > 500):
        warn('Warning: exp overflowing!', RuntimeWarning)
    return np.exp(np.minimum(v, 500))

def safe_log(v):
    """
    Safely apply np.log.  If the input is beyond a "safe" range, it will be clipped and a warning
    will be issued.

    """
    v = np.array(v)
    if np.any(v < -300):
        warn('Warning: exp overflowing!', RuntimeWarning)
    return np.log(np.maximum(v, -300))

def choice(num, total):
    """
    Returns indecies to (total) randomly selected items out of (num) without replacement.

    """
    return np.random.permutation(total)[:num]

def imshow(img, ax=None):
    """
    Displays an image, taking care of rescaling it and taking care of gray versus color.

    Parameters
    ----------
    img : numpy array, HxW or or HxWx1 or HxWx3 size
    ax : axes object    

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    prm = dict(interpolation='none')
    if img.ndim==2 or img.shape[2]==1:
        prm['cmap']=plt.cm.gray
        if img.ndim > 2:
            img = img[:,:,0]
    img -= img.min()
    img /= img.max()
    ax.imshow(img, **prm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def imshows(imgs, num_row=1):
    """
    Display a set of images.
    
    Parameters
    ----------
    imgs : HxWxN or HxWx3xN image stack
    num_row : number of rows in the grid of images
    
    """
    num_imgs = imgs.shape[-1]
    fig = plt.figure()
    num_col = int(np.ceil(float(num_imgs)/num_row))
    
    for i in xrange(num_imgs):
        ax = fig.add_subplot(num_row, num_col, i)
        imshow(imgs[...,i], ax=ax)