import tensorflow as tf
from tensorflow import glorot_normal_initializer
from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
from tensorflow.math import sqrt

import numpy as np
from numpy import random

class CONVGRU:
    def __init__(self, input_dimensions, output_size, num_layers, dtype=tf.float64):
        self.layers_hidden = [input_dimensions[-1],8,16,32,64]
        self.initializer = glorot_normal_initializer()
        self.kernel_size = [1,5,9,13]
        self.num_layers = num_layers
        self.output_size = output_size

        # initialize placeholders
        self.initialize_inputs(input_dimensions)

        # initialize params
        self.initialize_params(input_dimensions)

        # build layers and graph
        self.construct_graph()
        
        # opimizer
        self.create_optimizer()


    def initialize_inputs(self,input_dimensions):
        self.batch_size, self.timesteps, self.img_dim1, self.img_dim2, self.filter_size = input_dimensions
       
        self.hidden_state_0 = tf.placeholder(dtype=tf.float32,shape=[None,self.img_dim1-sum(self.kernel_size[1:])+self.num_layers, \
                                self.img_dim2-sum(self.kernel_size[1:])+self.num_layers,
                                self.layers_hidden[self.num_layers]])
 
        # Create a placeholder for model input
        self.input_data = tf.placeholder(dtype=tf.float32,shape=[None,None]+list(input_dimensions[2:]), name='input_data')
        
        # Create a placeholder for the expected output
        self.expected_output = tf.placeholder(dtype=tf.float32, shape=(None,None, self.output_size), name='expected_output')
    
    def create_optimizer(self):
        # Just use quadratic loss
        self.loss = tf.reduce_sum(0.5 * tf.pow(self.prediction_output - self.expected_output, 2)) \
                 / tf.cast(tf.shape(self.expected_output)[0], dtype=tf.float32)
        # loss function defined by softmax cross entropy
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.expected_output))

        # Use the Adam optimizer for training
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)


    def initialize_params(self, input_dimensions):
        self.batch_size, self.timesteps, self.img_dim1, self.img_dim2, self.filter_size = input_dimensions
 
        # filter weights for final hidden timestep at final layer
        self.W = {i:{element:tf.Variable(self.initializer([max(1,sum(self.kernel_size[i+1:])+1*(1+i-self.num_layers)), \
                                                           max(1,sum(self.kernel_size[i+1:])+1*(1+i-self.num_layers)), \
                                                           self.layers_hidden[i], \
                                                           self.layers_hidden[self.num_layers]])) \
                        for element in ['r','z','h']} for i in range(1,self.num_layers+1)}
        
        # filter weights for previous layer hidden variables
        self.S = {i:{element:tf.Variable(self.initializer([self.kernel_size[i],self.kernel_size[i], \
                                                           self.layers_hidden[i-1],self.layers_hidden[i]])) \
                        for element in ['r','z','h']} for i in range(1,self.num_layers+1)}

        # filter weights input
        self.V = {i:{element:tf.Variable(self.initializer([sum(self.kernel_size[1:i+1])-i+1,sum(self.kernel_size[1:i+1])-i+1, \
                                                           self.layers_hidden[0],self.layers_hidden[i]])) \
                        for element in ['r','z','h']} for i in range(1,self.num_layers+1)}
        
        # Biases for hidden states
        self.b = {i:{element:tf.Variable(self.initializer([self.img_dim1-sum(self.kernel_size[1:i+1])+i, \
                                                           self.img_dim2-sum(self.kernel_size[1:i+1])+i, \
                                                           self.layers_hidden[i]])) \
                        for element in ['r','z','h']} for i in range(1,self.num_layers+1)}

        self.Wout1 = tf.Variable(self.initializer([self.layers_hidden[self.num_layers],self.layers_hidden[self.num_layers]//2]))
        self.Wout2 = tf.Variable(self.initializer([self.layers_hidden[self.num_layers]//2,self.output_size]))
        self.bias1 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=(self.layers_hidden[self.num_layers]//2,), mean=0, stddev=0.01), name='b1')
        self.bias2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=(self.output_size,), mean=0, stddev=0.01), name='b2')


    def construct_graph(self):
        self.x_t = tf.transpose(self.input_data, perm=[1,0,2,3,4], name='x_t')
        self.h_t = tf.scan(self.forward_pass, self.x_t, initializer=self.hidden_state_0, name='h_t')
        self.output_frame = tf.scan(self.output_layer, self.h_t, initializer=self.initializer([-1,self.output_size]))

        self.h_t = tf.transpose(self.h_t, perm=[1,0,2,3,4])
        self.prediction_output = tf.transpose(self.output_frame, perm=[1,0,2])

    def output_layer(self, frame, hidden_state):
        hidden_ac = tf.nn.relu(hidden_state)
        hidden_pool = tf.nn.max_pool2d(hidden_ac,ksize=[1,self.img_dim1-sum(self.kernel_size[1:])+self.num_layers, \
                                                          self.img_dim2-sum(self.kernel_size[1:])+self.num_layers, 1], \
                                                strides=1,padding='VALID')
        hidden_squeeze = tf.squeeze(hidden_pool,axis=[1,2])
        out1 = tf.matmul(hidden_squeeze,self.Wout1) + self.bias1
        final_output = tf.matmul(out1,self.Wout2) + self.bias2
        
        return final_output

    def forward_pass(self, h_tm1, x_t):
        h_k = tf.zeros(tf.shape(self.hidden_state_0))
        #[self.hidden_state_0.shape.as_list()[0],self.img_dim1-sum(self.kernel_size[1:])+self.num_layers, \
        #                        self.img_dim2-sum(self.kernel_size[1:])+self.num_layers,
        #                        self.layers_hidden[self.num_layers]])
        for k in range(1,self.num_layers+1):
            h_k = self.gru_iterations(h_tm1, x_t, h_k, k)
        return h_k
    
    def gru_iterations(self, h_tm1, x_t, h_k, layer):
        # Definition of z_t
        output_shape = [tf.shape(self.input_data)[0]]+self.b[layer]['z'].get_shape().as_list()
        z_htm1 = tf.nn.conv2d_transpose(h_tm1,self.W[layer]['z'],output_shape,strides=1,padding='VALID',name='zconv2d_transpose_'+str(layer))
        z_x = tf.nn.conv2d(x_t,self.V[layer]['z'],strides=1,padding='VALID')
        if layer > 1:
            z_hk = tf.nn.conv2d(h_k,self.S[layer]['z'],strides=1,padding='VALID')
            z_t = tf.sigmoid(z_htm1 + z_x + z_hk + self.b[layer]['z'])
        else:
            z_t = tf.sigmoid(z_htm1 + z_x + self.b[layer]['z'])

        # Definition of r_t
        output_shape = [tf.shape(self.input_data)[0]]+self.b[layer]['r'].get_shape().as_list()
        r_htm1 = tf.nn.conv2d_transpose(h_tm1,self.W[layer]['r'],output_shape,strides=1,padding='VALID',name='rconv2d_transpose_'+str(layer))
        r_x = tf.nn.conv2d(x_t,self.V[layer]['r'],strides=1,padding='VALID')
        if layer > 1:
            r_hk = tf.nn.conv2d(h_k,self.S[layer]['r'],strides=1,padding='VALID')
            r_t = tf.sigmoid(r_htm1 + r_x + r_hk + self.b[layer]['r'])
        else:
            r_t = tf.sigmoid(r_htm1 + r_x + self.b[layer]['r'])
 

        # Definition of h~_t
        output_shape = [tf.shape(self.input_data)[0]]+self.b[layer]['h'].get_shape().as_list()
        h_htm1 = tf.nn.conv2d_transpose(h_tm1,self.W[layer]['h'],output_shape,strides=1,padding='VALID',name='hconv2d_transpose_'+str(layer))
        h_x = tf.nn.conv2d(x_t,self.V[layer]['h'],strides=1,padding='VALID')
        h_proposal = tf.tanh(h_x + tf.multiply(r_t, h_htm1) + self.b[layer]['h'])
        
        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_htm1) + tf.multiply(z_t, h_proposal)
        
        return h_t
