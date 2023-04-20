import os
import kaggle
import rasterio
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from patch_images import CropImages


class LakesDataset(Dataset):

    def __init__(self, data_path=None, download=False, patch_size=None, train=True, val=None):
        '''
        data_path is used if data directory is not the working directory
        download=True downloads data from Kaggle 
        patch_size=size (e.g. 256/512/1024...) if you want to use images of fixed size (croppes images into patches and saves in data_size/ folder if it's not done)
        train=False means that test dataset is going to be available
        '''

        self.data_path = data_path
        self.patch_size = patch_size

        if download:
            self.download_dataset(self.data_path)

        if self.patch_size:
            self.patch_images() 
            if self.data_path:
                self.data_path = self.data_path + 'data_' + str(self.patch_size)
            else:
                self.data_path = 'data_' + str(self.patch_size)

        else:
            self.data_path = 'data'
                               
            
        # The data should be structured in the working folder as {data (data_size): {train: [images, labels], test: [images, labels]}]}

        
        self.images_path = self.data_path + '/images/'
        self.labels_path = self.data_path + '/labels/'

        self.filenames_train = os.listdir(self.images_path)[:int(len(os.listdir(self.images_path))*0.8)]
        self.filenames_test = os.listdir(self.images_path)[int(len(os.listdir(self.images_path))*0.8):]


        if train and val:
            self.filenames = self.filenames_train[:int(len(self.filenames_train)*(1-val))]
        elif not train and val:
            self.filenames = self.filenames_train[int(len(self.filenames_train)*(1-val)):]
        elif train and not val:
            self.filenames = self.filenames_train
        else:
            self.filenames = self.filenames_test
    def download_dataset(self):

        ''' For this you need to have kaggle.json file in users/username/.kaggle folder (or wherever you specify it)
            This function may be extended if I ever have motivation for it, otherwise please unpack data manually in 'data/' folder '''
        
        kaggle.api.authenticate()
        path = self.data_path
        if self.data_path == '':
            path = None
        kaggle.api.dataset_download_files("elenagolimblevskaia/glacier-lakes-detection-via-satellite-images", path=path, quiet=False)

    def patch_images(self):
        if ('data_' + str(self.patch_size)) not in os.listdir(self.data_path):
            if self.data_path:
                im_crop = CropImages(self.data_path + 'data/', self.patch_size)
                im_crop.crop_images()
            else:
                im_crop = CropImages('data/', self.patch_size)
                im_crop.crop_images()



    def normalize(self, array):
        """Normalizes input arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))


    def __len__(self):
        '''Returns the total number of patches'''
        return len(self.filenames)
        

    def __getitem__(self, idx):
        
        '''In case of patching returns several images, otherwise one (but we are going to use patching since the images are too big)'''

        im_name = self.filenames[idx]
        lbl_name = self.filenames[idx] + 'f'

        # Reading images

        img = (rasterio.open(self.images_path + im_name))
        lbl = (rasterio.open(self.labels_path + lbl_name))

        img = np.array([self.normalize(img.read(1)), self.normalize(img.read(2)), self.normalize(img.read(3))], dtype=np.float32)
        lbl = np.array((lbl.read(1) > 0.5).astype(np.float32))
        
        return ToTensor()(img).permute(1,2,0), ToTensor()(lbl)

def get_loaders(patch_size, data_path=None, val_split=None, batch_size=None, shuffle=False):
    if val_split:
        train_dataset =LakesDataset(train=True, val=val_split, data_path=data_path, patch_size=patch_size)
        val_dataset = LakesDataset(train=False, val=val_split, data_path=data_path, patch_size=patch_size)
        test_dataset =LakesDataset(train=False, data_path=data_path, patch_size=patch_size)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

        return {'train_loader':train_loader, 'val_loader':val_loader, 'test_loader':test_loader}
    
    else:
        train_dataset =LakesDataset(train=True, val=val_split, data_path=data_path, patch_size=patch_size)
        test_dataset =LakesDataset(train=False, data_path=data_path, patch_size=patch_size)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

        return {'train_loader':train_loader, 'test_loader':test_loader}