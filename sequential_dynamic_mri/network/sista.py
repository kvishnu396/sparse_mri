# Author: Vishnu Kaimal
# Implement class structure and needed
# tools for CSISTA-RNN-MRI

import tensorflow as tf
from tensorflow import glorot_normal_initializer
from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
from tensorflow.math import sqrt

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
        self.conv_F = tf.Variable(glorot_normal_initializer([3,3,nch,32])), name = 'F'+str(layer_num))
        self.alpha = tf.Variable(glorot_normal_initializer([1])), name = 'alpha'+str(layer_num))
        self.lam_1 = tf.Variable(glorot_normal_initializer([1])), name = 'lam1'+str(layer_num))
        self.lam_2 = tf.Variable(glorot_normal_initializer([1])), name = 'lam2'+str(layer_num))

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
    def __init__(self, name, num_layers):
        
        self.name = name
        self.nch = nch
        self.K = num_layers
        self.filter_size = [3,3,nc,32]
        self.layers = []

        # initialize placeholders
        self.initialize_inputs()

        # initialize sista parameters
        self.create_sista_param()
        
        # build layers and graph
        self.create_graph()


    def create_graph(self):
        self.y_t = tf.transpose(self.input_img, [4,0,1,2,3], name='y_t')
        self.h_t = tf.scan(self.forward_pass, self.y_t, initializer=self.hidden_state_0, name='h_t')
        
        output_frame = tf.nn.conv2d(curr_hidden_state,self.conv_D,strides =[1,1,1,1],padding='SAME')


    def forward_pass(self, h_t, y_t):
        h_k = tf.zeros([None,nx,ny,nc])
        for i in range(1,self.K+1):
            h_k = self.sista_iterations(h_k, y_t, h_tm1, i)

        return h_k


    def sista_iterations(self, h_k, y_t, h_tm1, i):
        
        #construction of S
        D_hk = tf.nn.conv2d(h_k,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        ATA = self.measurement_operator(D_hk,self.mask)
        DTATA = tf.nn.conv2d_transpose(ATA,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        
        #construction of W
        D_ht = tf.nn.conv2d(h_tm1,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        F_D = tf.nn.conv2d(D_ht,self.conv_F[i],strides =[1,1,1,1],padding='SAME')
        P = tf.nn.conv2d_transpose(F_D,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        D_P = tf.nn.conv2d(P,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        ATADP = self.measurement_operator(D_P,self.mask)
        DTATADP = tf.nn.conv2d_transpose(ATADP,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        
        
        V = 1/self.alpha*tf.nn.conv2d_transpose(y_t,self.conv_D[i],strides =[1,1,1,1],padding='SAME')
        S = h_k- 1/alpha*DTATA + self.lam_2*D_hk
        if layer_num > 1:
            W = P*self.lam_2/self.alpha
        else:
            W = P*(self.alpha+self.lam2)/self.alpha - 1/self.alpha*DTATADP + self.lam2*D_P

        sum_activation = W + S + V
        
        curr_hidden_state = tf.nn.relu(sum_activation)

        return curr_hidden_state


    def measurement_operator(self,input_img,mask):
        batch, nx, ny, nc = input_img.shape
        image_permute = tf.transpose(input_img, perm=[0,3,1,2])
        kspace = 1/sqrt(nx*ny)*fftshift(fft2d(ifftshift(image_permute,axis=[2,3])),axis=[2,3])
        k_und = kspace*mask
        img_und = sqrt(nx*ny)*fftshift(fft2d(ifftshift(image_permute,axis=[2,3])),axis=[2,3])
        
        return img_und 
        
         

    def create_sista_param(self):
        self.hidden_state_0 = tf.Variable(glorot_normal_initializer([None,nx,ny,nc])), name = 'hidden_state_0')
        
        self.conv_D = self.conv_F = self.alpha = sel.lam_1 = self.lam_2 = {}
        #self.hidden_state = {0:self.hidden_state_0}
        
        for i in range(1,self.K+1):
            self.conv_D[i] = tf.Variable(glorot_normal_initializer(self.filter_size), name = 'D'+str(layer_num))
            self.conv_F[i] = tf.Variable(glorot_normal_initializer(self.filter_size)), name = 'F'+str(layer_num))
            self.alpha[i] = tf.Variable(glorot_normal_initializer([1])), name = 'alpha'+str(layer_num))
            self.lam_1[i] = tf.Variable(glorot_normal_initializer([1])), name = 'lam1'+str(layer_num))
            self.lam_2[i] = tf.Variable(glorot_normal_initializer([1])), name = 'lam2'+str(layer_num))

    def initialize_inputs(self):
        self.input_img = tf.placeholder(tf.float32, shape=(None,nx,ny,nc,nt))
        self.mask = tf.placeholder(tf.float32, shape=(None,nx,ny,nc,nt))
        self.gndtruth = tf.placeholder(tf.float32, shape=(None,nx,ny,nc,nt))

            
            



