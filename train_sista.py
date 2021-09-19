import os
import tensorflow as tf
import argparse
import numpy as np
from scipy.io import savemat, loadmat

#models
from sequential_dynamic_mri.network.sfista_mri import sista_mri

#training modules
from sequential_dynamic_mri.utils.train_test import train_test_module, prep_input
from sequential_dynamic_mri.utils.load_data import create_dataset
from sequential_dynamic_mri.utils.img2k import coil_combine, img2k, k2img
from sequential_dynamic_mri.utils.undersampling import var_dens_cartesian_mask, undersample
from sequential_dynamic_mri.plot.metric_graph import calc_mse, calc_psnr, calc_mssim

# params for testing/training
data_dir = '/home/ddev/sparse_mri/sparse_mri/OCMR/process_ocmr_data'
dataset_size = len(os.listdir(data_dir))
file_save = os.getcwd()
train_pct = 0.75
test_pct = 1 - train_pct
num_samples = dataset_size
num_epoch = 10
batch_size = 1
num_layers = 3
learning_rate = .001
save_dir = os.path.join('models', 'sista')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--acceleration_factor', metavar='int', nargs=1,
                        default=['4'],
                        help='Acceleration factor for k-space sampling')

args = parser.parse_args()
acc = int(args.acceleration_factor[0])  # undersampling rate
model_name = 'sista_' + str(acc)


# create training/testing data handler
train_test_set = train_test_module(dataset_size,data_dir)
train_test_set.create_test_train_split(train_pct,test_pct,num_samples)

# initalize session
session = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
model = sista_mri(model_name,num_layers,batch_size)
init_variables = tf.global_variables_initializer()
session.run(init_variables)

saver = tf.train.Saver()

# Initialize the losses
train_losses = []
validation_losses = []
train_mse = []
validation_mse = []
train_psnr = []
validation_psnr = []
train_mssim = []
validation_mssim = []


# save for testing with other methods
#test_batch_dict = train_test_set.create_test_dataset()
#savemat("raw_batch.mat", test_batch_dict)#{'img_batch':img_batch,'sen_batch':sen_batch,'k_batch':k_batch})

print("\n\n\n")
print("Start training....")
print()

# training sequence
for epoch in range(num_epoch):
    tot_train_loss = 0
    tot_test_loss = 0
    for batch_num, batch in enumerate(train_test_set.epoch_trainer(epoch,batch_size=batch_size)):
        img, sen, k = batch
        img_gnd, k_coil, mask, sen, k_und_coil, img_und_coil, img_und = prep_input(batch,acc) # prep input based on acceleration factor
       
        # Compute the losses
        train_recon_dmri, _, train_loss = session.run([model.recon_frame, model.train_step, model.complete_loss], \
                    feed_dict={model.input_kspace: k_und_coil, model.mask: mask, model.sensitivity: sen, model.gndtruth: img_gnd})
        tot_train_loss+=train_loss
        
        train_loss = 0.1
        train_recon_dmri = img_und
        train_losses.append(train_loss)
        train_mse.append(calc_mse(train_recon_dmri,img_gnd)[0])
        train_psnr.append(calc_psnr(train_recon_dmri,img_gnd)[0])
        train_mssim.append(calc_mssim(train_recon_dmri,img_gnd)[0])
       
        # testing sequence
        tot_test_loss = []
        tot_test_mse = []
        tot_test_psnr = []
        tot_test_mssim = []
        for j, batch in enumerate(train_test_set.test_generator()):
            img, sen, k = batch
            img_gnd, k_coil, mask, sen, k_und_coil, img_und_coil, img_und = prep_input(batch,acc) # prep input based on acceleration factor
            
            test_recon_dmri, validation_loss = session.run([model.recon_frame, model.complete_loss], \
                    feed_dict={model.input_kspace: k_und_coil, model.mask: mask, model.sensitivity: sen, model.gndtruth: img_gnd})

            tot_test_loss+=validation_loss
            
            validation_loss = 0.1
            test_recon_dmri = img_und
            tot_test_loss.append(validation_loss)
            tot_test_mse.append(calc_mse(test_recon_dmri,img_gnd)[0])
            tot_test_psnr.append(calc_psnr(test_recon_dmri,img_gnd)[0])
            tot_test_mssim.append(calc_mssim(test_recon_dmri,img_gnd)[0])

        validation_losses.append(np.mean(tot_test_loss))
        validation_mse.append(np.mean(tot_test_mse))
        validation_psnr.append(np.mean(tot_test_psnr))
        validation_mssim.append(np.mean(tot_test_mssim))
       

        print("Epoch {} Batch {} Training Loss: {}, MSE: {}, PSNR: {}, MSSIM: {}" \
                    .format(epoch, batch_num, train_loss, train_mse[-1], train_psnr[-1], train_mssim[-1]))

        print("Epoch {} Batch {} Testing Loss: {}, MSE: {}, PSNR: {}, MSSIM: {}" \
                    .format(epoch, batch_num, validation_losses[-1], validation_mse[-1], validation_psnr[-1], validation_mssim[-1]))
        
        print()

        
        #Periodically save the model.
        model_save_name = model_name + '_' + str(epoch*num_epoch+batch_num) + '.cptk'
        if epoch % 1 == 0:
            saver.save(session,os.path.join(save_dir,model_save_name))


