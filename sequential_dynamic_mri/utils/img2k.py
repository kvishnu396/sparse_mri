# Author: Vishnu Kaimal
# Provide functions to convert image domain to k-space and vice versa

from numpy.fft import *
from numpy import sqrt
import numpy as np
from scipy.linalg import dft
from copy import copy

# convert image to k-space
def img2k(img):
    dim_img = img.shape
    k = copy(img)
    if len(dim_img) == 2:
        return 1/sqrt(dim_img[0]*dim_img[1])*fftshift(fft2(ifftshift(img)))
    elif len(dim_img) == 3:
        for i in range(dim_img[-1]):
            frame = img[:,:,i]
            k[:,:,i] = 1/sqrt(dim_img[0]*dim_img[1])*fftshift(fft2(ifftshift(frame)))
    elif len(dim_img) == 4:
        for i in range(dim_img[-1]):
            for j in range(dim_img[-2]):
                frame = img[:,:,j,i]
                k[:,:,j,i] = 1/sqrt(dim_img[0]*dim_img[1])*fftshift(fft2(ifftshift(frame)))
    return k

# convert k-space to image domain
def k2img(k):
    dim_k = k.shape
    img = copy(k)
    if len(dim_k) == 2:
        return sqrt(dim_k[0]*dim_k[1])*fftshift(ifft2(ifftshift(k)))
    elif len(dim_k) == 3:
        for i in range(dim_k[-1]):
            k_frame = k[:,:,i]
            img[:,:,i] = sqrt(dim_k[0]*dim_k[1])*fftshift(ifft2(ifftshift(k_frame)))
    elif len(dim_k) == 4:
        for i in range(dim_k[-1]):
            for j in range(dim_k[-2]):
                k_frame = k[:,:,j,i]
                img[:,:,j,i] = sqrt(dim_k[0]*dim_k[1])*fftshift(ifft2(ifftshift(k_frame)))
    return img

# returns dft matrix
def dft_matrix(size):
    rows, cols = size
    if rows == cols:
        return dft(rows, scale='sqrtn')
    
    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    coeffs = np.outer(row_range, col_range)
    fourier_matrix = np.exp(coeffs * (-2. * np.pi * 1j / cols)) * scale

    return fourier_matrix

# image recombine with sensitivitiy coils
def coil_combine(img,sen,index_coil=3):
    size_img = img.shape
    size_sen = img.shape
    assert size_img == size_sen, "Size mismatch between img and sen!"
    
    recon = np.sum(img*np.conjugate(sen),index_coil)/np.sum(abs(sen)**2,index_coil)
    return recon


