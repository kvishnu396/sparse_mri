# Author: Vishnu Kaimal
# Functions for computing various reconstruction metrics

import numpy as np
import tensorflow as tf
from scipy import signal
#from skimage.metrics import structural_similarity

# calculation of mean squared for recon images
def mse(recon, ground_truth):
    return np.mean(np.abs(recon - ground_truth)**2)

# calculation of psnr for recon images
# peak signal to noise measure ratio of max power of signal
# to the power of noise
# peak: normalized or max
def psnr(recon, ground_truth, peak='max'):
    calc_mse = mse(recon, ground_truth)
    if peak == 'max':
        return 20*np.log10(np.amax(np.abs(ground_truth))/np.sqrt(calc_mse))
    else:
        return 20*np.log10(1/np.sqrt(calc_mse))


# calculation of ssim for recon images
# SSIM measures similarity between images
# measures 3 traits: luminance, contrast, structure
# 11x11 gaussian weighting matrix used for calculation
def ssim(recon, ground_truth):
    #recon = np.absolute(recon)
    #ground_truth = np.absolute(ground_truth)
    gaus = np.array([signal.windows.gaussian(11,std=1.5)])
    gaus_window = np.transpose(gaus)*gaus
   
    mu_r = signal.convolve2d(recon,gaus_window,mode="valid") 
    mu_g = signal.convolve2d(ground_truth,gaus_window,mode="valid")
    mu_r2 = mu_r**2
    mu_g2 = mu_g**2
    mu_gr = mu_r*mu_g
    sigma_r2 = signal.convolve2d(recon*recon,gaus_window,mode="valid") 
    sigma_g2 = signal.convolve2d(ground_truth*ground_truth,gaus_window,mode="valid") 
    sigma_rg = signal.convolve2d(recon*ground_truth,gaus_window,mode="valid")
    C1 = C2 = 0.05*np.max(np.absolute(ground_truth))
    #print(mu_r.shape, mu_g.shape, mu_r2.shape, mu_g2.shape, mu_gr.shape, sigma_r2.shape, sigma_g2.shape, sigma_rg.shape)
    
    ssim = (2*mu_r*mu_g+C1)*(2*sigma_rg+C2)/(mu_r2+mu_g2+C1)/(sigma_r2+sigma_g2+C2)
    ssim = np.abs(np.mean(ssim))
    #print(ssim.shape)
    return ssim

# mean ssim
# mean of 100 patches
def mssim(recon,ground_truth):
    num_patches = 100
    nx, ny = recon.shape
    dim_x = nx//10 + 1
    dim_y = ny//10 + 1
    tot_ssim = 0
    count = 0
    for i in range(nx//dim_x+1):
        for j in range(ny//dim_y+1):
            count+=1
            #print(recon[i*dim_x:(i+1)*dim_x,j*dim_y:(j+1)*dim_y].shape)
            #print(ground_truth[i*dim_x:(i+1)*dim_x,j*dim_y:(j+1)*dim_y].shape) 
            tot_ssim+= ssim(recon[i*dim_x:(i+1)*dim_x,j*dim_y:(j+1)*dim_y],ground_truth[i*dim_x:(i+1)*dim_x,j*dim_y:(j+1)*dim_y])
    return tot_ssim/count

    

    
    
    


