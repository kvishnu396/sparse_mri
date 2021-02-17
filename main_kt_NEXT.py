import torch
import numpy as np
import argparse, os
from torch.autograd import Variable
import torch.optim as optim
from scipy.io import loadmat
import time
import gc

# kt next modules
from ktnext.utils import compressed_sensing as cs
from ktnext.utils.metric import complex_psnr
from ktnext.network.kt_NEXT import *
from ktnext.utils.dnn_io import to_tensor_format
from ktnext.utils.dnn_io import from_tensor_format

#training modules
from sequential_dynamic_mri.utils.train_test import train_test_module
from sequential_dynamic_mri.utils.load_data import create_dataset
from sequential_dynamic_mri.utils.img2k import coil_combine, img2k, k2img
from sequential_dynamic_mri.utils.undersampling import var_dens_cartesian_mask, undersample

# params for testing/training
dataset_size = 5
file_save = os.getcwd()
train_pct = .75
test_pct = 1 - train_pct
num_samples = 40
num_epoch = 10
batch_size = 5
learning_rate = .001
save_dir = os.path.join('models', 'ktnext')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--acceleration_factor', metavar='int', nargs=1,
                        default=['4'],
                        help='Acceleration factor for k-space sampling')

args = parser.parse_args()
acc = int(args.acceleration_factor[0])  # undersampling rate
model_name = 'kt_NEXT_' + str(acc)



# prep input for training/testing
def prep_input(batch):
    img_batch, sen_batch, k_batch = batch
    num_samples, nx, ny, nc, nt = img_batch.shape
    img_coil = coil_combine(img_batch,sen_batch)
    k_coil = np.array([img2k(img_coil[i]) for i in range(num_samples)])
    msk = var_dens_cartesian_mask((nx,ny), acc, acc, slice_samp='horiz')
    msk = np.repeat(msk[:,:,np.newaxis],nt,axis=2)
    msk = np.repeat(msk[np.newaxis,:,:,:],num_samples,axis=0)
    k_und_coil = undersample(msk,k_coil)
    img_und_coil = np.array([k2img(k_und_coil[i]) for i in range(num_samples)])

    im_gnd_l = torch.from_numpy(to_tensor_format(img_coil,convert=False))
    im_und_l = torch.from_numpy(to_tensor_format(img_und_coil,convert=False))
    k_und_l = torch.from_numpy(to_tensor_format(k_und_coil,convert=False))
    mask_l = torch.from_numpy(to_tensor_format(msk,convert=False,mask=True))
    xf_gnd_l = torch.from_numpy(to_tensor_format(k_coil,convert=False))

    return im_und_l, k_und_l, mask_l, im_gnd_l, xf_gnd_l


# create training/testing data handler
train_test_set = train_test_module(dataset_size)
train_test_set.create_test_train_split(train_pct,test_pct,num_samples)
out = train_test_set.epoch_trainer(1,batch_size=batch_size)
#[print(element.shape) for element in out]
#out = train_test_set.test_generator()
#[print(element.shape) for element in out]


# cuda = True if torch.cuda.is_available() else False
cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

nc = 4
# build the model
xf_net = kt_NEXT_model(nc=nc)
criterion = torch.nn.MSELoss()
if cuda:
    xf_net = xf_net.cuda()
    criterion.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, xf_net.parameters()), lr=learning_rate, betas=(0.5, 0.999))
pytorch_total_params = sum(p.numel() for p in xf_net.parameters() if p.requires_grad)
print('Starting training!')
print('Total trainable params: %d' % pytorch_total_params)



for epoch in range(0, num_epoch+1):
    gc.collect()
    t_start = time.time()
    train_err = 0
    train_batches = 0

    for element in train_test_set.epoch_trainer(epoch,batch_size=batch_size):
        x_und, k_und, mask, x_gnd, xf_gnd = prep_input(element)
        print(x_und.shape,k_und.shape,mask.shape,x_gnd.shape,xf_gnd.shape)
        x_u = Variable(x_und.type(Tensor))
        k_u = Variable(k_und.type(Tensor))
        mask = Variable(mask.type(Tensor))
        gnd = Variable(x_gnd.type(Tensor))
        xf_gnd = Variable(xf_gnd.type(Tensor))

        optimizer.zero_grad()
        xf_out, img = xf_net(x_u, k_u, mask)

        loss = criterion(img['t%d' % (nc - 1)], gnd) + criterion(xf_out['t%d' % (nc-1)], xf_gnd)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xf_net.parameters(), 5)
        optimizer.step()

        train_err += loss.item()
        train_batches += 1

        t_end = time.time()
        train_err /= train_batches

        xf_net.eval()
        test_loss = []
        base_psnr = []
        epoch_psnr = []
    
        print("Testing!")
        for element in train_test_set.test_generator():#epoch_trainer(epoch,batch_size=batch_size):
            x_und, k_und, mask, x_gnd, xf_gnd = prep_input(element)#im, acc)
            x_u = Variable(x_und.type(Tensor))
            k_u = Variable(k_und.type(Tensor))
            mask = Variable(mask.type(Tensor))
            gnd = Variable(x_gnd.type(Tensor))

            with torch.no_grad():
                xf_out, img = xf_net(x_u, k_u, mask)

            test_loss.append(criterion(img['t%d' % (nc-1)], gnd).item())

            im_und = from_tensor_format(x_und.numpy())
            im_gnd = from_tensor_format(x_gnd.numpy())
            im_rec = from_tensor_format(img['t%d' % (nc-1)].data.cpu().numpy())

            for idx in range(im_und.shape[0]):
                base_psnr.append(complex_psnr(im_gnd[idx], im_und[idx]))
                epoch_psnr.append(complex_psnr(im_gnd[idx], im_rec[idx]))

        print("Epoch {}/{}".format(epoch + 1, train_batches))#num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        print(" testing loss:\t\t{:.6f}".format(np.mean(test_loss)))
        print(" base PSNR:\t\t{:.6f}".format(np.mean(base_psnr)))
        print(" test PSNR:\t\t{:.6f}".format(np.mean(epoch_psnr)))

        name = model_name#'model_epoch_%d.npz' % epoch
        torch.save(xf_net.state_dict(), os.path.join(save_dir, name))
        print('model parameters saved at %s' % os.path.join(save_dir, name))
        print('')
