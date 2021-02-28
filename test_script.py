# Author: Vishnu Kaimal
# Test functionality of various modules

import numpy as np
#from sequential_dynamic_mri.utils import test
from sequential_dynamic_mri.utils.undersampling import poly_pdf
from sequential_dynamic_mri.utils.img2k import coil_combine
from sequential_dynamic_mri.utils.load_data import create_dataset

p = 1/8
img_shape = [409, 409]
poly_power = 8
slice_samp = 'horiz'
poly_pdf(p,img_shape,poly_power,slice_samp)
 
img_batch, sen_batch, _ = create_dataset(1)

img = img_batch[0]
sen = sen_batch[0]

comb = coil_combine(img,sen,index_coil=2)
print(comb.shape)
print('high:',np.max(np.abs(comb)))

