# Author: Vishnu Kaimal
# define function to plot individual frames of DMRI

import matplotlib.pyplot as plt
import numpy as np


def plot_frames(batch_dmri, batch_index, frames, save_loc, label, recon_type):
    num_samples, nx, ny, nt = batch_dmri.shape
    assert batch_index<= num_samples, "Index for batch out of range!"
    assert max(frames) < nt, "Frame timestamp out of range!"

    batch_dmri = batch_dmri[batch_index]
    for i in frames:
        curr_frame = np.abs(batch_dmri[:,:,i])
        high = np.amax(curr_frame)
        low = np.amin(curr_frame)
        fname = save_loc + recon_type + '/' + label + '_' + str(i)+ '.png'
        plt.imsave(fname,curr_frame,vmin=low,vmax=high)
        #plt.savefig(save_loc+label+'.png',bbox_inches='tight')
        #plt.clf()
        
        
    
