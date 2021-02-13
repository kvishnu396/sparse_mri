# Author: Vishnu Kaimal
# Create class for creating train/test data sets for model

directory = '/home/ddev/sparse_mri/sequential_dynamic_mri/cine_data/'

from .load_data import index_cine_batch, create_batch  
from random import sample, shuffle

# class structure for smooth train/testing sets for model
class train_test_module(object):
    def __init__(self):
        self.index_mat_files, self.index_par_files = index_cine_batch(directory)
        self.train_index = self.test_index = []

    # create appropriate test/train split
    def create_test_train_split(self,train=0, test=1, num_samples=None):
        num_samples = len(self.index_mat_files) if not num_samples else num_samples
        assert test+train == 1, "Test plus train probabilities must equal 1"
        assert num_samples<=len(self.index_mat_files), "Number of samples greater than data set size!"

        self.num_test_samples = int(num_samples*test)
        self.num_train_samples = num_samples - self.num_test_samples
        
        self.train_index = sample(list(self.index_mat_files.keys()),self.num_train_samples)
        left_over_index = [element for element in list(self.index_mat_files.keys()) if element not in self.train_index]
        self.test_index = sample(left_over_index, self.num_test_samples)

    # create epoch generator
    def epoch_trainer(self, epoch, batch_size=5):
        shuffle(self.train_index)
        num_iterations = len(self.train_index)//batch_size + 1
        for i in range(num_iterations):
            batch_index = self.train_index[batch_size*i:batch_size*(i+1)]
            print("Created training set: Epoch %d Batch %d" %(epoch,i))
            yield create_batch(self.index_mat_files, batch_index)

    # test data generator
    def test_generator(self, batch_size=5):
        num_iterations = len(self.test_index)//batch_size+1
        print("Preparing Test data!")
        for i in range(num_iterations):
            batch_index = self.test_index[batch_size*i:batch_size*(i+1)]
            yield create_batch(self.index_mat_files, batch_index)
        
        
        
    


    



        
        

        
        
        
        
        
        
        





