#import requests
#import StringIO
from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile
import os
import os.path

data_dir = os.path.join('.', 'data', 'immutable')

class DLProgress(tqdm):
    """
    Class is taken from helper.py file provided
    for Udacity Deep Learning Foundations Project 2
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
        
def download_traffic_sign_data(data_dir, force):
    """
    Download the traffic signs data
    """
    file_was_downloaded = False
    zip_file_name = 'traffic-signs-data.zip'
    zip_file_path = os.path.join(data_dir, zip_file_name)
    
    # create directory to save to if not exists
    if not os.path.isdir(data_dir):
        user_id = os.geteuid()
        group_id = os.getegid()
    
        os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)
        os.chown(data_dir, user_id, group_id)
    
    # download file is not exists, or if force download requested
    if not os.path.isfile(zip_file_path) or force:
        url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip'
        
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='German Traffic Sign Dataset') as pbar:
            urlretrieve(url, zip_file_path, pbar.hook)
            
        file_was_downloaded = True
        
    return zip_file_path, file_was_downloaded
                
def get_traffic_signs_data(force=False):
    """
    Will download zip file if not already present.
    Will download zip file if force==True
    
    Unzips files in zip to './data/immutable'
    """
    # download zip file 
    zip_file_path, file_was_downloaded = download_traffic_sign_data(data_dir, force)
    
    # zip already exists so download was skipped
    if file_was_downloaded == False:
        print("File download skipped, it already exists, use force==True to overwrite.\n".format(data_dir))
    
    # extract pickled files
    if file_was_downloaded == True or force == True:
        extracted_file_paths = []
        print("Unzipping files...".format(data_dir))
        
        with zipfile.ZipFile(zip_file_path) as f:
            for name in tqdm(f.namelist()):
                f.extract(name, data_dir)
                extracted_file_paths.append(os.path.join(data_dir, name))

        print('Files extracted:')
        for extracted_file_path in extracted_file_paths:
            print(extracted_file_path)
    else:
        print("Skipping unzipping of pickled files, as they've been previously extracted,\nuse force==True to overwrite")
    
    # used to return file paths for the pickled files to be later loaded
    training_file_path = os.path.join(data_dir, 'train.p')
    validation_file_path = os.path.join(data_dir, 'valid.p')
    testing_file_path = os.path.join(data_dir, 'test.p')
    
    return training_file_path, validation_file_path, testing_file_path
