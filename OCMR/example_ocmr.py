import pandas


# NOTE: Change this to the S3 location when finalized
ocmr_data_attributes_location = './ocmr_data_attributes.csv'

df = pandas.read_csv('./ocmr_data_attributes.csv')
# Cleanup empty rows and columns
df.dropna(how='all', axis=0, inplace=True)
df.dropna(how='all', axis=1, inplace=True)

# Show the first 10 items in the list
print(df.head(10))
print()
print(df)

selected_df = df.query ('`file name`.str.contains("fs_") and slices=="1"',engine='python')# and scn=="15avan" and viw=="lax"', engine='python')
print(selected_df)


import boto3
from botocore import UNSIGNED
from botocore.client import Config

import os

# The local path where the files will be downloaded to
download_path = './ocmr_data'

# Replace this with the name of the OCMR S3 bucket 
bucket_name = 'ocmr'

if not os.path.exists(download_path):
    os.makedirs(download_path)
    
count=1
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Iterate through each row in the filtered DataFrame and download the file from S3. 
# Note: Test after finalizing data in S3 bucket
for index, row in selected_df.iterrows():
    print('Downloading {} to {} (File {} of {})'.format(row['file name'], download_path, count, len(selected_df)))
    s3_client.download_file(bucket_name, 'data/{}'.format(row['file name']), '{}/{}'.format(download_path,row['file name']))
    count+=1



#read ocmr data
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('Python')
from ismrmrdtools import show, transform
# import ReadWrapper
import read_ocmr as read

# Load the data, display size of kData and scan parmaters
filename = './ocmr_data/fs_0012_3T.h5'    # fs_0007_1_5T.h5'
kData,param = read.read_ocmr(filename);
print('Dimension of kData: ', kData.shape,len(kData.shape))


quit()
# Image reconstruction (SoS)
dim_kData = kData.shape; CH = dim_kData[3]; SLC = dim_kData[6]; 
kData_tmp = np.mean(kData, axis = 8); # average the k-space if average > 1
print(kData_tmp.shape,len(kData_tmp.shape))

im_coil = transform.transform_kspace_to_image(kData_tmp, [0,1]); # IFFT (2D image)
im_sos = np.sqrt(np.sum(np.abs(im_coil) ** 2, 3)); # Sum of Square
print('Dimension of Image (with ReadOut ovesampling): ', im_sos.shape)
RO = im_sos.shape[0];
image = im_sos[math.floor(RO/4):math.floor(RO/4*3),:,:]; # Remove RO oversampling
print('Dimension of Image (without ReadOout ovesampling): ', image.shape)



