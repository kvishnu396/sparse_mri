# Author: Vishnu Kaimal
# Functions to load/process .mat files

from scipy.io import loadmat, savemat
import multiprocessing as mp
import numpy as np
import time as sleeper
import os
import pickle

from .img2k import img2k, k2img

directory = '/home/ddev/sparse_mri/sequential_dynamic_mri/cine_data/'

# indexing files training/test set based on appropriate parameters
# cardiac cine samples are mapped to index values for retrieval
def index_cine_batch(directory):
    folders_cine = os.listdir(directory)
    mat_file_names = []
    mat_par_names = []
    for folder in folders_cine:
        try:
            files = [element for element in os.listdir(directory+folder) if '.mat' in element]
            files = [directory+folder+'/'+element for element in files]
            [mat_par_names.append(element) if '_par' in element else mat_file_names.append(element) for element in files]
        except NotADirectoryError:
            pass
    index_mat_files = {i:file_name for i, file_name in enumerate(mat_file_names)}
    index_par_files = {i:file_name for i, file_name in enumerate(mat_par_names)}
    
    return index_mat_files, index_par_files

# function for individual process of file list for batch
def create_batch(index_mat_files, batch_index):
    img_batch = sen_batch = k_batch = None
    print("Creating batch with index set",batch_index,"...")
    for index in batch_index:
        filename = index_mat_files[index]
        img, sen, k_data = load_cine_file(index, filename)
        if type(img_batch) == type(None):
            img_batch = np.array([img])
            sen_batch = np.array([sen])
            k_batch = np.array([k_data])
        else:
            img_batch = np.concatenate((img_batch, np.array([img])))
            sen_batch = np.concatenate((sen_batch, np.array([sen])))
            k_batch = np.concatenate((k_batch, np.array([k_data])))
    print(img_batch.shape, sen_batch.shape, k_batch.shape)
    return img_batch, sen_batch, k_batch
    
# load individual cine files and retrieve image and coil sensitivities
def load_cine_file(index, filename):
    print("Index: ", index, "Loading File: ", filename)
    start = sleeper.time()
    data = loadmat(filename)   # load cine .mat file
    
    # retrieve data
    print(data.keys())    
    img = data['img']
    sen = data['sen']
    #msk = data['msk']
    if 'k_data' in data:
        k_data = data['k_data']
    else:
        k_data = img2k(img)
        savemat(filename,{'img':img,'sen':sen,'k_data':k_data},do_compression=True)
    
    print("Time for loading: ", index, "is ", sleeper.time()-start) 
    return img, sen, k_data

# aggregate cine samples
def create_dataset(file_save,num_proc=10):
    index_mat_files, _ = index_cine_batch(directory)
    data_list = [i for i,element in index_mat_files.items()][:5]
    len_group = len(data_list)//num_proc + 1
    parts = [(index_mat_files, data_list[i:i+len_group]) for i in range(0,len(data_list), len_group)]
    pool = mp.Pool(processes=num_proc)
    out = pool.starmap(create_batch, parts)
    img_batch = sen_batch = k_batch = None
    for img, sen, k_data in out:
        if type(img_batch) == type(None):
            img_batch = img#np.array([img])
            sen_batch = sen#np.array([sen])
            k_batch = k_data#np.array([k_data])
        else:
            img_batch = np.concatenate((img_batch, img))#np.array([img])))
            sen_batch = np.concatenate((sen_batch, sen))#np.array([sen])))
            k_batch = np.concatenate((k_batch, k_data))#np.array([k_data])))
    print(img_batch.shape,sen_batch.shape,k_batch.shape)
    savemat(file_save+"raw_batch.mat",{'img_batch':img_batch,'sen_batch':sen_batch,'k_batch':k_batch})
   
    return img_batch, sen_batch, k_batch

#test loading and processing images
def test_load():
    index_mat_files, index_par_files = index_cine_batch(directory)
    for index, filename in index_mat_files.items():
        img, sen, ksp = load_cine_file(index, filename)
        print([img.shape,sen.shape,ksp.shape])
    print("Loaded all cine_files")

# load param
def load_cine_par(filename):
    pass

if __name__ == "__main__":
    file_name = 'cine_data.pickle'
    start = sleeper.time()
    test_load()
    print(sleeper.time()-start)
    #pickle.dump(out,open(file_name,"wb"))
