# Written by: Vishnu Kaimal
# Script to extract OCMR data

import pandas
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

ocmr_data_attributes_location = './ocmr_data_attributes.csv'
download_path = './extract_ocmr_data'
bucket_name = 'ocmr'

# get table of ocmr data attributes
df = pandas.read_csv(ocmr_data_attributes_location)
df.dropna(how='all', axis=0, inplace=True)
df.dropna(how='all', axis=1, inplace=True)


selected_df = df.query ('`file name`.str.contains("fs_") and slices=="1"',engine='python')# and scn=="15avan" and viw=="lax"', engine='python')


if not os.path.exists(download_path):
    os.makedirs(download_path)
    
count=1
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# iterate through selected df and download file from S3
for index, row in selected_df.iterrows():
    print('Downloading {} to {} (File {} of {})'.format(row['file name'], download_path, count, len(selected_df)))
    s3_client.download_file(bucket_name, 'data/{}'.format(row['file name']), '{}/{}'.format(download_path,row['file name']))
    count+=1



