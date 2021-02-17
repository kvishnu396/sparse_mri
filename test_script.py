# Author: Vishnu Kaimal
# Test functionality of various modules

#from sequential_dynamic_mri.utils import test
from sequential_dynamic_mri.utils.undersampling import poly_pdf

p = 1/8
img_shape = [409, 409]
poly_power = 8
slice_samp = 'horiz'
poly_pdf(p,img_shape,poly_power,slice_samp)
 

