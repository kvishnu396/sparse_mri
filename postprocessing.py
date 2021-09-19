# Author: Vishnu Kaimal
# python script to evaluate each reconstruction methods

import numpy as np
from scipy.io import savemat, loadmat
from sequential_dynamic_mri.utils.img2k import coil_combine, img2k, k2img
from sequential_dynamic_mri.utils.score_metrics import mssim, ssim
from sequential_dynamic_mri.plot.plotframe import plot_frames
from sequential_dynamic_mri.plot.error_plot import plot_error
from sequential_dynamic_mri.plot.metric_graph import plot_metrics


# params for all postprocessing
frames = [5,10,15,20]
batch_index = 0
save_loc = '/home/ddev/sparse_mri/results/postproc/'
save_recon = '/home/ddev/sparse_mri/results/recon/'


#raw_cine = loadmat('raw_batch.mat') 
raw_cine_us = loadmat('raw_batch_undersampled.mat') 

# coil combine to establish ground truth
# plot frames of ground truth
##########################################################################
#raw_cine = loadmat('raw_batch.mat') 
img_batch = raw_cine_us['img_batch']
sen_batch = raw_cine_us['sen_batch']
num_samples, nx, ny, nc, nt = img_batch.shape
gndtruth = []

for i in range(num_samples):
    gndtruth.append(coil_combine(img_batch[i,:,:,:,:],sen_batch[i,:,:,:,:],index_coil=2))
gndtruth = np.array(gndtruth)

index = 10
print(mssim(gndtruth[0,:,:,index],gndtruth[0,:,:,index]))
print(ssim(gndtruth[0,:,:,index],gndtruth[0,:,:,index]))

label = "gndtruth"
plot_frames(gndtruth, batch_index, frames, save_loc, label, label)
print("Constructed Ground Truth!")
print("Created frame figures for Ground Truth", frames)

########################################################################## 


# 4 times undersampling
##########################################################################
print("Beginning 4 times undersampling...")
recon_dict = {}
k_4 = raw_cine_us['k_4']
img_4 = np.array([k2img(k_4[i]) for i in range(num_samples)])
recon_dict['under_4'] = ('undersample',np.array([coil_combine(img_4[i],sen_batch[i],index_coil=2) for i in range(num_samples)]))

recon_dict['LS_4'] = ('L+S',loadmat(save_recon+'LS_4.mat')['LS_4'])
recon_dict['grouse_4'] = ('GROUSE',loadmat(save_recon+'grouse_4.mat')['grouse_4'])
recon_dict['sistamri_4'] = ('SISTA-MRI',loadmat(save_recon+'sistamri_4.mat')['sistamri_4'])

for label, val in recon_dict.items():
    recon_type, recon_img = val
    error_label = label + '_error'
    plot_frames(recon_img, batch_index, frames, save_loc, label, recon_type)
    plot_error(recon_img, gndtruth, batch_index, frames, save_loc, error_label, recon_type)

print("Plotted frames and error image for 4 times undersampling!")

plot_metrics(recon_dict, gndtruth, 4, save_loc)

# 8 times undersampling
##########################################################################
print("Beginning 8 times undersampling...")
recon_dict = {}
k_8 = raw_cine_us['k_8']
img_8 = np.array([k2img(k_8[i]) for i in range(num_samples)])
recon_dict['under_8'] = ('undersample',np.array([coil_combine(img_8[i],sen_batch[i],index_coil=2) for i in range(num_samples)]))

recon_dict['LS_8'] = ('L+S',loadmat(save_recon+'LS_8.mat')['LS_8'])
recon_dict['grouse_8'] = ('GROUSE',loadmat(save_recon+'grouse_8.mat')['grouse_8'])
recon_dict['sistamri_8'] = ('SISTA-MRI',loadmat(save_recon+'sistamri_8.mat')['sistamri_8'])


for label, val in recon_dict.items():
    recon_type, recon_img = val
    error_label = label + '_error'
    plot_frames(recon_img, batch_index, frames, save_loc, label, recon_type)
    plot_error(recon_img, gndtruth, batch_index, frames, save_loc, error_label, recon_type)

print("Plotted frames and error image for 8 times undersampling!")

plot_metrics(recon_dict, gndtruth, 8, save_loc)

# 12 times undersampling
##########################################################################
print("Beginning 12 times undersampling...")
recon_dict = {}
k_12 = raw_cine_us['k_12']
img_12 = np.array([k2img(k_12[i]) for i in range(num_samples)])
recon_dict['under_12'] = ('undersample',np.array([coil_combine(img_12[i],sen_batch[i],index_coil=2) for i in range(num_samples)]))

recon_dict['LS_12'] = ('L+S',loadmat(save_recon+'LS_12.mat')['LS_12'])
recon_dict['grouse_12'] = ('GROUSE',loadmat(save_recon+'grouse_12.mat')['grouse_12'])
recon_dict['sistamri_12'] = ('SISTA-MRI',loadmat(save_recon+'sistamri_12.mat')['sistamri_12'])


for label, val in recon_dict.items():
    recon_type, recon_img = val
    error_label = label + '_error'
    plot_frames(recon_img, batch_index, frames, save_loc, label, recon_type)
    plot_error(recon_img, gndtruth, batch_index, frames, save_loc, error_label, recon_type)

print("Plotted frames and error image for 12 times undersampling!")

plot_metrics(recon_dict, gndtruth, 12, save_loc)
















