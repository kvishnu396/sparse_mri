# Author: Vishnu Kaimal
# python script to evaluate each reconstruction methods

import numpy as np
from scipy.io import savemat, loadmat
from sequential_dynamic_mri.utils.img2k import coil_combine, img2k, k2img
from sequential_dynamic_mri.plot.plotframe import plot_frames
from sequential_dynamic_mri.plot.error_plot import plot_error
from sequential_dynamic_mri.plot.metric_graph import plot_metrics


# params for all postprocessing
frames = [5,10,15,20]
batch_index = 0
save_loc = '/home/ddev/sparse_mri/results/postproc/'
save_recon = '/home/ddev/sparse_mri/results/recon/'


raw_cine = loadmat('raw_batch.mat') 
raw_cine_us = loadmat('raw_batch_undersampled.mat') 

# coil combine to establish ground truth
# plot frames of ground truth
##########################################################################
raw_cine = loadmat('raw_batch.mat') 
img_batch = raw_cine['img_batch']
sen_batch = raw_cine['sen_batch']
num_samples, nx, ny, nc, nt = img_batch.shape
gndtruth = []

for i in range(num_samples):
    gndtruth.append(coil_combine(img_batch[i,:,:,:,:],sen_batch[i,:,:,:,:],index_coil=2))
gndtruth = np.array(gndtruth)

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
#recon_dict['csistarnnmri_4'] = None

for label, val in recon_dict.items():
    recon_type, recon_img = val
    error_label = label + '_error'
    plot_frames(recon_img, batch_index, frames, save_loc, label, recon_type)
    plot_error(recon_img, gndtruth, batch_index, frames, save_loc, label, recon_type)

print("Plotted frames and error image for 4 times undersampling!")

plot_metrics(recon_dict, gndtruth, 4, save_loc)


















