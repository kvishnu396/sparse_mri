# Author: Vishnu Kaimal
# Implement class structure and needed
# tools for CSISTA-RNN-MRI

import tensorflow as tf
from tensorflow import glorot_normal_initializer
from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
from tensorflow.math import sqrt
import numpy as np

nx = 409
ny = 409
nc = 2
nt = 24

# one iteration of sista block
class sista_block(object):
    def __init__(self,input_frame,hidden_state_t,hidden_state_k,mask,nch,layer_num):
        super(sista_block, self).__init__()

        # initialize sista parameters
        self.conv_D = tf.Variable(glorot_normal_initializer([3,3,nch,32]), name = 'D'+str(layer_num))
        self.conv_F = tf.Variable(glorot_normal_initializer([3,3,nch,32]), name = 'F'+str(layer_num))
        self.alpha = tf.Variable(glorot_normal_initializer([1]), name = 'alpha'+str(layer_num))
        self.lam_1 = tf.Variable(glorot_normal_initializer([1]), name = 'lam1'+str(layer_num))
        self.lam_2 = tf.Variable(glorot_normal_initializer([1]), name = 'lam2'+str(layer_num))

        #construction of S
        D_hk = tf.nn.conv2d(hidden_state_k,self.conv_D,strides =[1,1,1,1],padding='SAME')
        ATA = self.measurement_operator(D_hk,mask)
        DTATA = tf.nn.conv2d_transpose(ATA,self.conv_D,strides =[1,1,1,1],padding='SAME')
        
        #construction of W
        D_ht = tf.nn.conv2d(hidden_state_t,self.conv_D,strides =[1,1,1,1],padding='SAME')
        F_D = tf.nn.conv2d(D_ht,self.conv_D,strides =[1,1,1,1],padding='SAME')
        P = tf.nn.conv2d_transpose(F_D,self.conv_D,strides =[1,1,1,1],padding='SAME')
        D_P = tf.nn.conv2d(P,self.conv_D,strides =[1,1,1,1],padding='SAME')
        ATADP = self.measurement_operator(D_P,mask)
        DTATADP = tf.nn.conv2d_transpose(ATADP,self.conv_D,strides =[1,1,1,1],padding='SAME')
        
        
        V = 1/self.alpha*tf.nn.conv2d_transpose(input_frame,self.conv_D,strides =[1,1,1,1],padding='SAME')
        S = hidden_state_k - 1/alpha*DTATA + self.lam_2*D_hk
        if layer_num > 1:
            W = P*self.lam_2/self.alpha
        else:
            W = P*(self.alpha+self.lam2)/self.alpha - 1/self.alpha*DTATADP + self.lam2*D_P

        sum_activation = W + S + V
        
        curr_hidden_state = tf.nn.relu(sum_activation)

        output_frame = tf.nn.conv2d(curr_hidden_state,self.conv_D,strides =[1,1,1,1],padding='SAME')

        return output_frame, curr_hidden_state 


    def measurement_operator(self,input_img,mask):
        batch, nx, ny, nc = input_img.shape
        image_permute = tf.transpose(input_img, perm=[0,3,1,2])
        kspace = 1/sqrt(nx*ny)*fftshift(fft2d(ifftshift(image_permute,axis=[2,3])),axis=[2,3])
        k_und = kspace*mask
        img_und = sqrt(nx*ny)*fftshift(fft2d(ifftshift(image_permute,axis=[2,3])),axis=[2,3])
        
        return img_und 

        
# creation of sista-rnn mri class
class sista_mri(object):
    def __init__(self, name, num_layers, batch_size):
        
        self.name = name
        self.nch = nc
        self.K = num_layers
        self.batch_size = batch_size
        self.filter_size = [3,3,nc,nc]#32]
        self.layers = []
        self.initializer = glorot_normal_initializer()

        # initialize placeholders
        self.initialize_inputs()

        # initialize sista parameters
        self.create_sista_param()

        # build layers and graph
        self.create_graph()

        #optimizer
        self.create_optimizer()

    def create_optimizer(self):
        #self.loss = self.gndtruth-self.recon_frame
        self.loss = tf.math.reduce_mean(tf.pow(tf.math.abs(self.gndtruth-self.recon_frame),2))
        print(self.loss.shape)
        # loss function defined by softmax cross entropy
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.expected_output))

        # Use the Adam optimizer for training
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def create_graph(self):
        self.y_t = self.conv_realcomplex(self.input_img)
        self.y_t = tf.transpose(self.y_t, perm=[4,0,1,2,3], name='y_t')
        self.h_t = tf.scan(self.forward_pass, self.y_t, initializer=self.hidden_state_0, name='h_t')
        self.output_frame = tf.scan(self.output_layer, self.h_t)
        
        self.h_t = tf.transpose(self.h_t, perm=[1,2,3,4,0])
        self.recon_frame = self.conv_realcomplex(tf.transpose(self.output_frame, perm=[1,2,3,4,0]))
        

    def output_layer(self, frame, hidden_state):
        return tf.nn.conv2d(hidden_state,self.conv_D[self.K],strides =[1,1,1,1],padding='SAME')

    def forward_pass(self, h_tm1, y_t):
        h_k = tf.zeros([self.batch_size,nx,ny,nc])
        for k in range(1,self.K+1):
            h_k = self.sista_iterations(h_k, y_t, h_tm1, k)

        return h_k


    def sista_iterations(self, h_k, y_t, h_tm1, i):
        
        #construction of S
        D_hk = tf.nn.conv2d(h_k,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        ATA = self.measurement_operator(D_hk,self.mask)
        DTATA = tf.nn.conv2d_transpose(ATA,self.conv_D[i],output_shape=[self.batch_size,nx,ny,nc],strides =[1,1,1,1],padding='SAME',name='DTATA')
        
        #construction of W
        D_ht = tf.nn.conv2d(h_tm1,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        F_D = tf.nn.conv2d(D_ht,self.conv_F[i],strides =[1,1,1,1],padding='SAME')
        P = tf.nn.conv2d_transpose(F_D,self.conv_D[i],output_shape=[self.batch_size,nx,ny,nc],strides =[1,1,1,1],padding='SAME',name='testP')
        D_P = tf.nn.conv2d(P,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        ATADP = self.measurement_operator(D_P,self.mask)
        DTATADP = tf.nn.conv2d_transpose(ATADP,self.conv_D[i],output_shape=[self.batch_size,nx,ny,nc],strides =[1,1,1,1],padding='SAME',name='DTATADP')
        
        
        print(y_t.shape)
        V = 1/self.alpha[i]*tf.nn.conv2d_transpose(y_t,self.conv_D[i],output_shape=[self.batch_size,nx,ny,nc],strides =[1,1,1,1],padding='SAME',name='V')
        S = h_k- 1/self.alpha[i]*DTATA + self.lam_2[i]*D_hk
        if i > 1:
            W = P*self.lam_2[i]/self.alpha[i]
        else:
            W = P*(self.alpha[i]+self.lam_2[i])/self.alpha[i] - 1/self.alpha[i]*DTATADP + self.lam_2[i]*D_P

        sum_activation = W + S + V
        
        bias = tf.repeat(self.lam_1[i]/self.alpha[i],2)
        curr_hidden_state = tf.nn.relu(tf.nn.bias_add(sum_activation,bias))

        return curr_hidden_state

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

    def measurement_operator(self,input_img,mask):
        batch, nx, ny, nc = input_img.shape
        image_permute = tf.transpose(input_img, perm=[0,3,1,2])
        img_complex = tf.complex(image_permute[:,0,:,:],image_permute[:,1,:,:])
        multiplier = 1/np.sqrt(nx.value*ny.value)
        kspace = multiplier*fftshift(fft2d(ifftshift(img_complex,axes=[1,2])),axes=[1,2])
        k_und = kspace*mask
        img_und = 1/multiplier*fftshift(fft2d(ifftshift(kspace,axes=[1,2])),axes=[1,2])
        real_img_und = tf.expand_dims(tf.math.real(img_und),1)
        imag_img_und = tf.expand_dims(tf.math.imag(img_und),1)
        img_und = tf.concat([real_img_und, imag_img_und],1)
        img_und = tf.transpose(img_und,perm=[0,2,3,1])
        
        return img_und 
        
         

    def create_sista_param(self):
        self.hidden_state_0 = tf.Variable(self.initializer([self.batch_size,nx,ny,nc]), name = 'hidden_state_0')
        
        self.conv_D = {}
        self.conv_F = {}
        self.alpha = {}
        self.lam_1 = {}
        self.lam_2 = {}
        #self.hidden_state = {0:self.hidden_state_0}
        for i in range(1,self.K+1):
            layer_num = i
            init_val_filter = self.initializer(self.filter_size)
            init_val_scaler = self.initializer((1,))
            self.conv_D[i] = tf.Variable(init_val_filter, name = 'D'+str(layer_num))
            self.conv_F[i] = tf.Variable(init_val_filter, name = 'F'+str(layer_num))
            self.alpha[i] = tf.Variable(self.initializer(shape=()), name = 'alpha'+str(layer_num))
            self.lam_1[i] = tf.Variable(self.initializer(shape=()), name = 'lam1'+str(layer_num))
            self.lam_2[i] = tf.Variable(self.initializer(shape=()), name = 'lam2'+str(layer_num))
        

    def initialize_inputs(self):
        self.input_img = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny,nt))
        self.mask = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny))
        self.gndtruth = tf.placeholder(tf.complex64, shape=(self.batch_size,nx,ny,nt))

            
            



