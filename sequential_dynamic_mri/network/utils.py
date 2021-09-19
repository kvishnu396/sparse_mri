# Author: Vishnu Kaimal
# Utility functions used for network model


import tensorflow as tf
from tensorflow import glorot_normal_initializer
from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
import numpy as np

# soft thresholding operator with relu
def soft_thresh(input_state, soft_thresh_param):
    rel_input = tf.nn.relu(input_state-soft_thresh_param)
    sign = tf.sign(input_state)
    return tf.math.multiply(sign,rel_input)

# learned sparsity transform
class sparse_transform(object):
    def __init__(self,nch,nf,input_dim,hidden_size,batch_size):
        super(sparse_transform, self).__init__()

        self.initializer = glorot_normal_initializer()
        
        # convolutional weights for transform
        self.conv_A = tf.Variable(self.initializer(shape=(5,5,nch,nf)), name = 'A')
        self.conv_B = tf.Variable(self.initializer(shape=(5,5,nch,nf)), name = 'B')

        self.transform_dim, self.inverse_dim, self.shape_a = self.calculate_dim_transform(input_dim,nf,batch_size)
        self.shape_input = [batch_size]+list(input_dim)+[nch]
        
        # forward transform
        self.Wf = tf.Variable(self.initializer(shape=(self.transform_dim,hidden_size)), name = 'Wf')
        self.bf = tf.Variable(self.initializer(shape=(hidden_size,)), name = 'bf')

        # inverse transform
        self.Wb = tf.Variable(self.initializer(shape=(hidden_size,self.transform_dim)), name = 'Wb')
        self.bb = tf.Variable(self.initializer(shape=(self.transform_dim,)), name = 'bb')
        
        # convolutional weights for inverse
        self.conv_C = tf.Variable(self.initializer(shape=(5,5,nch,nf)), name = 'C')
        self.conv_D = tf.Variable(self.initializer(shape=(5,5,nch,nf)), name = 'D')


    def calculate_dim_transform(self,input_dim,nf,batch_size):
        dim_x, dim_y = input_dim
        return (dim_x-8)*(dim_y-8)*nf, (batch_size,dim_x-8,dim_y-8,nf),(batch_size,dim_x-4,dim_y-4,nf)

    def left_inverse_loss(self,blank,input_frame):
        return self.inverse_transform(self.transform(input_frame))

    def transform_coil(self,blank,input_frame):
        return self.transform(input_frame)

    # sparse transform operation
    def transform(self,input_frame):
        # first composite
        a = tf.nn.conv2d(input_frame,self.conv_A,strides=1,padding='VALID')
        rel_a = tf.nn.relu(a)
        
        # second composite
        b = tf.nn.conv2d(rel_a,self.conv_B,strides=1,padding='VALID')
        rel_b = tf.nn.relu(b)

        # flatten output
        flatten_b = tf.layers.Flatten()(rel_b)
    
        fc_hidden = tf.matmul(flatten_b,self.Wf) + self.bf

        return fc_hidden

    # inverse sparse sparse transform
    def inverse_transform(self,hidden_layer):
        fc_hidden = tf.matmul(hidden_layer,self.Wb) + self.bb
        
        # reshape input
        rhidden = tf.reshape(fc_hidden, shape=self.inverse_dim)
        
        # first composite
        c = tf.nn.conv2d_transpose(rhidden,self.conv_C,self.shape_a,strides=1,padding='VALID')
        rel_c = tf.nn.relu(c)

        # second composite
        d = tf.nn.conv2d_transpose(rel_c,self.conv_D,self.shape_input,strides=1,padding='VALID')
        rel_d = tf.nn.relu(d)

        return rel_d

# measurement operator
class measurement_operator(object):
    def __init__(self,sensitivity, mask):
        super(measurement_operator, self).__init__()
        self.sensitivity = sensitivity
        self.mask = mask

    # measurement operator with mask
    def measurement_operator(self,input_img):
        batch, nx, ny, _ = input_img.shape
        image_permute = tf.transpose(input_img, perm=[0,3,1,2])
        sen_permute = tf.transpose(self.sensitivity, perm=[3,0,1,2])
        img_complex = tf.complex(image_permute[:,0,:,:],image_permute[:,1,:,:])
        coil_sen = tf.transpose(img_complex*sen_permute, perm=[1,2,3,0])
        multiplier = 1/np.sqrt(nx.value*ny.value)
        kspace = multiplier*fftshift(fft2d(ifftshift(coil_sen,axes=[1,2])),axes=[1,2])
        kspace = tf.transpose(kspace, perm=[3,0,1,2])
        k_und_coil = tf.transpose(self.mask*kspace, perm=[1,2,3,0])
        
        return k_und_coil

    # inverse measurement operator
    def inverse_measurement(self,kspace):
        batch, nx, ny, _ = kspace.shape
        multiplier = np.sqrt(nx.value*ny.value)
        kspace = tf.transpose(kspace, perm=[3,0,1,2])
        k_und_coil = tf.transpose(self.mask*kspace, perm=[1,2,3,0])
        img_und = multiplier*fftshift(fft2d(ifftshift(k_und_coil,axes=[1,2])),axes=[1,2])
        real_img_und = tf.expand_dims(tf.math.real(img_und),1)
        imag_img_und = tf.expand_dims(tf.math.imag(img_und),1)
        img_und = tf.concat([real_img_und, imag_img_und],1)
        dmri_img = tf.transpose(img_und,perm=[0,2,3,1,4])

        return dmri_img

        

