# Written by: Vishnu Kaimal
# main script to process ocmr data
# extraction, coil estimation, and processing

# Before running the code, install ismrmrd-python and ismrmrd-python-tools:
#  https://github.com/ismrmrd/ismrmrd-python
#  https://github.com/ismrmrd/ismrmrd-python-tools

import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os
from ismrmrdtools import show, transform
import read_ocmr as read
import coil_est
from scipy.io import savemat, loadmat


ocmr_dir = 'extract_ocmr_data'
save_dir = 'process_ocmr_data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# extract ocmr
print("Extracting OCMR data...")
import extract_ocmr # runs python extract ocmr script

# processing and coil estimation
# (nx,ny,nc,nt) are dimensions of processed ocmr
print()
print("Processing OCMR data...")
ocmr_files = os.listdir(ocmr_dir)
dict_dim = {}
for ocmr_file in ocmr_files:
    ocmr_filename = os.path.join(ocmr_dir, ocmr_file)
    kData, param = read.read_ocmr(ocmr_filename)
    kData = np.mean(kData, axis=8)
    kData = np.squeeze(kData)
    dim = kData.shape[:2]
    if dim != (384,144):
        continue
    if dim in dict_dim:
        dict_dim[dim]+=1
    else:
        dict_dim[dim]=1
    img = transform.transform_kspace_to_image(kData, [0,1])
    index_file = os.path.join(save_dir,ocmr_file.split(".")[0]+".mat")
    sens = coil_est.smoothed_coil_estimation(img)
    output_dict = {"k_data": kData, "img": img, "sen":sens}
    savemat(index_file, output_dict)
    print(index_file, kData.shape,img.shape,sens.shape)

print(dict_dim)




