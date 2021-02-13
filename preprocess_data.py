# Author: Vishnu Kaimal
# Preprocess data for train/test
# Load data and perform undersampling

import multiprocessing as mp
import numpy as np
import time
from sequential_dynamic_mri.utils.load_data import create_dataset
from sequential_dynamic_mri.utils.undersampling import var_dens_cartesian_mask, undersample
from scipy.io import savemat, loadmat
from numpy.matlib import repmat


####Variable density sampling param######
acc = 4
poly_power = 4
slice_samp_param = 'vert'


file_save = '/home/ddev/sparse_mri/'

start = time.time()
# create raw batch data
img_batch, sen_batch, k_batch = create_dataset(file_save)
print(img_batch.shape,sen_batch.shape,k_batch.shape)
print("Time Taken: ",time.time()-start)

# complete undersampling
num_samples, nx, ny, nc, nt = k_batch.shape
accs = [4,8,12] # acceleration rates
msk_dict = {}

for acc in accs:
    start = time.time()
    msk = var_dens_cartesian_mask((nx,ny), acc, acc, slice_samp='horiz')
    print(msk.shape)
    msk = np.repeat(msk[:,:,np.newaxis],nc,axis=2)
    print(msk.shape)
    msk = np.repeat(msk[:,:,:,np.newaxis],nt,axis=3)
    print(msk.shape)
    msk = np.repeat(msk[np.newaxis,:,:,:,:],num_samples,axis=0)
    print(msk.shape)
    print("Time Taken: ",time.time()-start)
    msk_dict[acc] = undersample(msk,k_batch)
    print(msk_dict[acc].shape)


savemat("raw_batch_undersampled.mat",{'img_batch':img_batch,'sen_batch':sen_batch,'k_batch':k_batch, \
                                        'k_4':msk_dict[4],'k_8':msk_dict[8],'k_12':msk_dict[12]})

data = loadmat("raw_batch_undersampled.mat")
print(data.keys())




