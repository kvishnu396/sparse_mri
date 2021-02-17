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
def psnr(recon, ground_truth, peak='normalized'):
    calc_mse = mse(recon, ground_truth)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(ground_truth))**2/mse)
    else:
        return 10*np.log10(1./mse)


# calculation of ssim for recon images
# SSIM measures similarity between images
# measures 3 traits: luminance, contrast, structure
# 11x11 gaussian weighting matrix used for calculation
def ssim(recon, ground_truth):
    gaus = np.array([signal.windows.gaussian(11,std=1.5)])
    gaus_window = np.transpose(gaus)*gaus
   
    mu_r = signal.convolve2d(recon,gaus_window,mode="same") 
    mu_g = signal.convolve2d(ground_truth,gaus_window,mode="same")
    mu_r2 = mu_r**2
    mu_g2 = mu_g**2
    mu_gr = mu_r*mu_g
    sigma_r2 = signal.convolve2d(recon*recon,gaus_window,mode="same") 
    sigma_g2 = signal.convolve2d(ground_truth*ground_truth,gaus_window,mode="same") 
    sigma_rg = signal.convolve2d(recon*ground_truth,gaus_window,mode="same")
    C1 = C2 = 0.05*np.max(np.absolute(ground_truth))
    
    ssim = (2*mu_r*mu_g+C1)*(2*sigma_rg+C2)/(mu_r2+mu_g2+C1)/(sigma_r2+sigma_g2+C2)
    
    return ssim

# mean ssim
def mssim(recon,ground_truth):
    num_frames = recon.shape[-1]
    tot_ssim = 0
    for i in range(num_frames):
        tot_ssim+= ssim(recon[:,:,i],ground_truth[:,:,i])
    return tot_ssim/num_frames

    

    
    
    


