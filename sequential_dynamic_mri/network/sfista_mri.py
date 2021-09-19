# Author: Vishnu Kaimal
# Implement class structure and needed
# tools for unrolled smoothed FISTA for MRI

import tensorflow as tf
from tensorflow import glorot_normal_initializer
from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
from tensorflow.math import sqrt
import numpy as np

from .utils import sparse_transform, measurement_operator, soft_thresh

nx = 384
ny = 144
nc = 2
nt = 24
nf = 32
alpha = beta = 0.01
hidden_size = 50


# creation of fsista unrolled sista-rnn mri class
class sista_mri(object):
    def __init__(self, name, num_layers, batch_size):
        
        self.name = name
        self.nch = nc
        self.K = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.filter_size = [3,3,nc,nc]
        self.layers = []
        self.initializer = glorot_normal_initializer()

        # initialize placeholders
        self.initialize_inputs()

        self.sparse_transform = sparse_transform(nc,nf,(nx,ny),self.hidden_size,batch_size)
        self.measurement_operator = measurement_operator(self.sensitivity, self.mask)
        
        # initialize sista parameters
        self.create_sista_param()

        # build layers and graph
        self.create_graph()

        #optimizer
        self.create_optimizer()

    def create_optimizer(self):
        # compute recon loss
        self.recon_loss = tf.math.reduce_mean(tf.pow(tf.math.abs(self.gndtruth-self.recon_frame),2))
        
        # compute transform loss
        real_recon_frame = tf.transpose(self.conv_realcomplex(self.recon_frame),perm=[4,0,1,2,3])
        transform_app = tf.transpose(tf.scan(self.sparse_transform.left_inverse_loss,real_recon_frame),perm=[1,2,3,4,0])
        transform_app = self.conv_realcomplex(transform_app)
        squared_err = tf.pow(tf.norm(transform_app - self.recon_frame,ord='fro',axis=(1,2),keepdims=True),2)
        squared_err = tf.abs(squared_err)
        sparse_prop = tf.norm(tf.scan(self.sparse_transform.transform_coil,real_recon_frame,initializer=self.hidden_state_0),ord=1)

        self.sparse_loss = alpha*tf.reduce_sum(squared_err) + beta*tf.reduce_sum(sparse_prop)

        self.complete_loss = self.recon_loss + self.sparse_loss

        # Use the Adam optimizer for training
        self.train_step = tf.train.AdamOptimizer().minimize(self.complete_loss)
        
    def create_graph(self):
        measurements = tf.transpose(self.input_kspace, perm=[4,0,1,2,3], name='measurements')
        self.h_t = tf.scan(self.forward_pass, measurements, initializer=self.hidden_state_0, name='h_t') # iterate through layers
        
        self.recon_hidden_0 = self.sparse_transform.inverse_transform(self.hidden_state_0)
        self.recon_frame = tf.scan(self.output_layer, self.h_t, initializer=self.recon_hidden_0) # iterate through time
        
        # final step
        self.h_t = tf.transpose(self.h_t, perm=[1,2,0])
        self.recon_frame = tf.transpose(self.recon_frame, perm=[1,2,3,4,0])
        self.recon_frame = self.conv_realcomplex(self.recon_frame)

    def output_layer(self, blank, hidden_state):
        return self.sparse_transform.inverse_transform(hidden_state)

    def forward_pass(self, h_tm1, m_t):
        z_k = self.sparse_transform.transform(tf.nn.conv2d(self.sparse_transform.inverse_transform(h_tm1),self.conv_F,strides=1,padding='SAME'))
        h_k1 = self.sparse_transform.transform(tf.nn.conv2d(self.sparse_transform.inverse_transform(h_tm1),self.conv_F,strides=1,padding='SAME'))
        for k in range(1,self.K+1):
            z_k, h_k1 = self.sista_iterations(h_k1, z_k, m_t, h_tm1, k)

        return h_k1


    def sista_iterations(self, h_k1, z_k, m_t, h_tm1, k):
        # compute gradient f(z)
        img_inverse_output = self.sparse_transform.inverse_transform(z_k) # inverse ouput (b,nx,ny,2)
        sensitivity_encoded = self.measurement_operator.measurement_operator(img_inverse_output) # sensitivity encoded recon
        residual_measurement = m_t - sensitivity_encoded
        undo_sensitivity = self.measurement_operator.inverse_measurement(residual_measurement) # inverse measurement
        undo_sensitivity = tf.transpose(undo_sensitivity,perm=[4,0,1,2,3])
        del_f_coil = tf.scan(self.sparse_transform.transform_coil,undo_sensitivity,initializer=self.hidden_state_0) # compute scan across coils
        del_f = tf.math.reduce_sum(del_f_coil,axis=0)
        
        # compute gradient g1
        del_g1 = h_k1 - soft_thresh(h_k1,self.lam_1[k]*self.mu[k])
        del_g1 = 1/self.mu[k]*(del_g1)

        # compute gradient g2
        img_tm1 = self.sparse_transform.inverse_transform(h_tm1)
        est_imgtk = tf.nn.conv2d(img_tm1, self.conv_F, strides=1, padding='SAME')
        img_tk1 = self.sparse_transform.inverse_transform(h_k1)
        temporal_sparse = tf.nn.conv2d(img_tk1 - est_imgtk, self.conv_W, strides=1, padding='SAME')
        del_g2 = temporal_sparse - soft_thresh(temporal_sparse,self.lam_2[k]*self.mu[k])
        del_g2 = tf.nn.conv2d(del_g2, self.conv_W, strides=1, padding='SAME')
        del_g2 = 1/self.mu[k]*self.sparse_transform.transform(del_g2)
        
        # compute h_tk
        sum_grad = del_f + del_g1 + del_g2
        h_tk = z_k - 1/self.L[k]*sum_grad
        
        # compute z_kp1
        z_kp1 = h_tk + self.rho[k]*(h_tk - h_k1)
        
        return z_kp1, h_tk
        
        
    def conv_realcomplex(self,input_space,dimension=3):
        if input_space.dtype in [tf.float32,tf.float64]:
            image_permute = tf.transpose(input_space, perm=[0,3,1,2,4])
            img_complex = tf.complex(image_permute[:,0,:,:,:],image_permute[:,1,:,:,:])
            return img_complex
        elif input_space.dtype in [tf.complex64, tf.complex128]:
            real_img = tf.expand_dims(tf.math.real(input_space),dimension)
            imag_img = tf.expand_dims(tf.math.imag(input_space),dimension)
            img_real = tf.concat([real_img, imag_img],dimension)
            return img_real

    def create_sista_param(self):
        self.hidden_state_0 = tf.Variable(self.initializer(shape=(self.batch_size,self.hidden_size)), name = 'hidden_state_0')
       
        init_val_filter = self.initializer(self.filter_size)
        init_val_scaler = self.initializer((1,))
 
        self.conv_F = tf.Variable(init_val_filter, name = 'F')
        self.conv_W = tf.Variable(init_val_filter, name = 'W')

        self.rho = {}
        self.lam_1 = {}
        self.lam_2 = {}
        self.mu = {}
        self.L = {}
        for layer_num in range(1,self.K+1):
            self.rho[layer_num] = tf.Variable(self.initializer(shape=()), name = 'alpha'+str(layer_num))
            self.lam_1[layer_num] = tf.Variable(self.initializer(shape=()), name = 'lam1'+str(layer_num))
            self.lam_2[layer_num] = tf.Variable(self.initializer(shape=()), name = 'lam2'+str(layer_num))
            self.mu[layer_num] = tf.Variable(self.initializer(shape=()), name = 'mu'+str(layer_num))
            self.L[layer_num] = tf.Variable(self.initializer(shape=()), name = 'L'+str(layer_num))


    def initialize_inputs(self):
        self.input_kspace = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny,None,None)) # (b,nx,ny,nc,nt)
        self.mask = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny))
        self.sensitivity = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny,None)) # (b,nx,ny,nc) 
        self.gndtruth = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny,None)) # (b,nx,ny,nt)

            
            



