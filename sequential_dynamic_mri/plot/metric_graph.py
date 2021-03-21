# Author: Vishnu Kaimal
# Create module for metric plots

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ..utils.score_metrics import mse, psnr, mssim, ssim

# function to calculate and plot all score metrics
# between ground truth and reconstructed image
# inputs: dict of reconmethods, ground truth, acc rate, save_loc

mse_dict = {}
psnr_dict = {}
ssim_dict = {}

def calc_mse(recon_img, gndtruth):
    num_samples, nx, ny, nt = gndtruth.shape
    mse_error = np.zeros((num_samples,nt))
    for i in range(num_samples):
        for j in range(nt):
            mse_error[i,j] = mse(recon_img[i,:,:,j],gndtruth[i,:,:,j])
    return np.mean(mse_error), np.mean(mse_error,axis=0)

def calc_psnr(recon_img, gndtruth):
    num_samples, nx, ny, nt = gndtruth.shape
    psnr_error = np.zeros((num_samples,nt))
    for i in range(num_samples):
        for j in range(nt):
            psnr_error[i,j] = psnr(recon_img[i,:,:,j],gndtruth[i,:,:,j])
    return np.mean(psnr_error), np.mean(psnr_error,axis=0)

def calc_mssim(recon_img, gndtruth):
    num_samples, nx, ny, nt = gndtruth.shape
    mssim_error = np.zeros((num_samples,nt))
    for i in range(num_samples):
        for j in range(nt):
            mssim_error[i,j] = ssim(recon_img[i,:,:,j],gndtruth[i,:,:,j])
    return np.mean(mssim_error), np.mean(mssim_error,axis=0)

def create_plot(error_dict, metric_type, save_loc):
    matplotlib.use('Agg')
    for recon_type, val in error_dict.items():
        tot_mean, time_mean = val
        plt.plot(range(1,len(time_mean)+1),time_mean, label=recon_type)
        print(recon_type + ": " + str(tot_mean))
    plt.xlabel('Frame')
    ylabel = metric_type+"(dB)" if metric_type == 'PSNR' else metric_type
    plt.ylabel(ylabel)
    #plt.title('')
    plt.legend()
    plt.savefig(save_loc, bbox_inches = 'tight')
    plt.clf()


def plot_metrics(recon_dict, gndtruth, acc, save_loc):
    save_loc+='metrics/'
    mse_dict = {}
    psnr_dict = {}
    ssim_dict = {}
    num_samples, nx, ny, nt = gndtruth.shape
    for label, val in recon_dict.items():
        recon_type, recon_img = val
        mse_dict[recon_type] = calc_mse(recon_img, gndtruth)
        psnr_dict[recon_type] = calc_psnr(recon_img, gndtruth)
        ssim_dict[recon_type] = calc_mssim(recon_img, gndtruth)

    print("MSE Metrics for " + str(acc) + " times undersampling")
    save_mse = save_loc + 'mse_' + str(acc) + '.png'
    create_plot(mse_dict,'MSE',save_mse)

    print("PSNR Metrics for " + str(acc) + " times undersampling")
    save_mse = save_loc + 'psnr_' + str(acc) + '.png'
    create_plot(psnr_dict,'PSNR',save_mse)

    print("MSSIM Metrics for " + str(acc) + " times undersampling")
    save_mse = save_loc + 'mssim_' + str(acc) + '.png'
    create_plot(ssim_dict,'MSSIM',save_mse)
