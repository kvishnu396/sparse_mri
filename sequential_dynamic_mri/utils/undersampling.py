# Author: Vishnu Kaimal
# Functions for undersampling k-space data

import numpy as np
from numpy.random import binomial, rand
import numpy.matlib

# undersample k space or image space with given mask
def undersample(msk, img):
    msk_shape = msk.shape
    img_shape = img.shape
    assert msk_shape == img_shape

    return msk*img


# application of uniform cartesian mask to 2D input
# bernouli probability of sampling based on acc rate
def uniform_cartesian_mask(img_shape, acc, slice_samp='horiz'):
    #acc -> acceleration rate(D/N) where D is size of input and N is num of samples
    #num_slices -> vert, horiz, both
    
    x_shape, y_shape = img_shape
    p = 1/acc

    if slice_samp == 'horiz':
        y_samples = binomial(1,p,y_shape)
        msk = np.matlib.repmat(y_samples,x_shape,1)
    elif slice_samp == 'vert':
        x_samples = np.transpose([binomial(1,p,x_shape)])
        msk = np.matlib.repmat(x_samples,1,y_shape)
    elif slice_samp == 'both':
        msk = binomial(1,p,img_shape)
    
    return msk

# variable density cartesian mask to 2D input
# first described by Lustig et. al where center frequencies
# have higher probability than those frequencies at ends
# polynomial or gaussian variable density sampling
def var_dens_cartesian_mask(img_shape, acc, dist_param, dist='polynomial', slice_samp='horiz'):
    #acc -> acceleration rate(D/N) where D is size of input and N is num of samples
    #dist_param -> parameter for power of polynomial or variance of gaussian
    #dist -> distribution used(polynomial or gaussian)
    #slice_samp -> vert, horiz, both

    x_shape, y_shape = img_shape
    p = 1/acc
    if dist == 'polynomial':
        pdf = poly_pdf(p,img_shape,dist_param,slice_samp)
    elif dis == 'gaussian':
        pdf = gaussian_pdf(p,img_shape,dist_param,slice_samp)

    if slice_samp == 'horiz':
        y_samples = rand(y_shape)<pdf
        y_samples = y_samples.astype(int)
        msk = np.matlib.repmat(y_samples,x_shape,1)
    elif slice_samp == 'vert':
        x_samples = rand(x_shape)<pdf
        x_samples = x_samples.astype(int)
        msk = np.matlib.repmat(x_samples,1,y_shape)
    elif slice_samp == 'both':
        msk = rand(x_shape,y_shape) < pdf
        msk = msk.astype(int)
    
    return msk

# complete iterative polynomial distribution
def poly_pdf(p,img_shape,poly_power,slice_samp):
    x_shape, y_shape = img_shape
    
    threshold_prob = int(p*x_shape*y_shape) if slice_samp == 'both' else int(p*x_shape) if slice_samp == 'vert' \
                            else int(p*y_shape) if slice_samp == 'horiz' else 0

    if slice_samp == 'both':
        x, y = np.meshgrid(np.linspace(-1,1,y_shape),np.linspace(-1,1,x_shape))
        r = np.maximum(np.abs(x),np.abs(y))
    elif slice_samp == 'vert':
        r = np.abs(np.linspace(-1,1,x_shape))
    elif slice_samp == 'horiz':
        r = np.abs(np.linspace(-1,1,y_shape))

    updf = (1-r)**poly_power
    minval = 0
    maxval = 1
    while True:
        val_thresh = minval/2 + maxval/2
        pdf = updf + val_thresh
        pdf[pdf>1] = 1 
        N = int(np.sum(pdf))
        if N > threshold_prob:
            maxval = val_thresh
        elif N < threshold_prob:
            minval = val_thresh
        elif N == threshold_prob:
            break
    return pdf

# calculation of gaussian pdf for input image
def gaussian_pdf(p,img_shape,sigma,slice_samp):
    x_shape, y_shape = img_shape
    
    if slice_samp == 'horiz':
        pdf = normal_pdf(y_shape,simga) 
    elif slice_samp == 'vert':
        pdf = normal_pdf(x_shape,sigma)
    elif slice_samp == 'both':
        vert_pdf = normal_pdf(x_shape,sigma)
        horiz_pdf = normal_pdf(y_shape,sigma)
        pdf = np.outer(ver_pdf,horiz_pdf)

    return pdf


# normal pdf based on image length and sima
def normal_pdf(length, sigma):
    const = 1/sgima/np.sqrt(2*np.pi)
    return const*np.exp(-1/2 * ((np.arange(length) - length / 2)/sigma)**2)


# sampling density used in kt FOCUSS
# code sourced from kt-next project
def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask

# sheer grid undersampling
# code sourced from Deep MRI project
def shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                    centred=False, sample_n=10):
    '''
    Creates undersampling mask which samples in sheer grid

    Parameters
    ----------

    shape: (nt, nx, ny)

    acceleration_rate: int

    Returns
    -------

    array

    '''
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in xrange(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep

