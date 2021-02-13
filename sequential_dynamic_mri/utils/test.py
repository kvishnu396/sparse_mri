# Author: Vishnu Kaimal
# Script to test various functionality

import numpy as np

# undersampling masks
from .undersampling import uniform_cartesian_mask
vert_out = uniform_cartesian_mask((5,7),4,'vert')
horiz_out = uniform_cartesian_mask((5,7),4,'horiz')
both_out = uniform_cartesian_mask((5,7),4,'both')
print("Random Cartesian Vert:\n",vert_out)
print("Random Cartesian horiz:\n",horiz_out)
print("Random Cartesian both:\n",both_out)

from .undersampling import var_dens_cartesian_mask
vert_out = var_dens_cartesian_mask((100,150), 4, 4, slice_samp='horiz')
horiz_out = var_dens_cartesian_mask((100,150), 4, 4, slice_samp='vert')
both_out = var_dens_cartesian_mask((100,150), 4, 4, slice_samp='both')
print("Variable Dense Cartesian Vert:",np.sum(vert_out)/100/150)
print("Variable Dense Cartesian horiz:",np.sum(horiz_out)/100/150)
print("Variable Dense Cartesian both:",np.sum(both_out)/100/150)

from .train_test import train_test_module

train_test_set = train_test_module()
train_test_set.create_test_train_split(.55,.45)
out = train_test_set.epoch_trainer(1)
[print(element.shape) for element in out]
out = train_test_set.test_generator()
[print(element.shape) for element in out]




