# Author: Vishnu Kaimal
# define function to plot error plot for frames of DMRI

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_error(batch_dmri, batch_gndtruth, batch_index, frames, save_loc, label, recon_type):
    num_samples, nx, ny, nt = batch_dmri.shape
    assert batch_index<= num_samples, "Index for batch out of range!"
    assert batch_dmri.shape == batch_gndtruth.shape, "Reconstruction and ground truth dimensions don't match!"
    assert max(frames) < nt, "Frame timestamp out of range!"

    batch_dmri = batch_dmri[batch_index]
    batch_gndtruth = batch_gndtruth[batch_index]
    for i in frames:
        curr_frame = batch_dmri[:,:,i]
        gnd_frame = batch_gndtruth[:,:,i]
        diff_frame = np.abs(gnd_frame-curr_frame)
        high = np.amax(diff_frame)
        low = np.amin(diff_frame)
        fname = save_loc + recon_type + '/' + label + '_' + str(i)+ '.png'
        plt.imsave(fname,diff_frame,vmin=low,vmax=high,cmap='gray')
        #plt.savefig(save_loc+label+'.png',bbox_inches='tight')
        #plt.clf()
        
        
    
