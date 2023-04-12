import os
import kaggle
import rasterio
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def download_dataset(data_path):

    ''' For this you need to have kaggle.json file in users/username/.kaggle folder (or wherever you specify it)
        This function may be extended if I ever have motivation for it, otherwise please unpack data manually '''
    
    kaggle.api.authenticate()
    path = data_path
    if data_path == '':
        path = None
    kaggle.api.dataset_download_files("elenagolimblevskaia/glacier-lakes-detection-via-satellite-images", path=path, quiet=False)
            

class LakesDataset(Dataset):

    def __init__(self, data_path='', download=False, transforms=None, train=True, patch_size=None):


        if download:
            download_dataset(data_path)

        self.path = data_path
        self.patch_size = patch_size
        self.device = get_device()

        # The data should be structured in the working folder as {data: {train: [images, labels], test: [images, labels]}]}

        if train:
            self.images_path = self.path + 'data/train/images/'
            self.labels_path = self.path + 'data/train/labels/'

        else: 
            self.images_path = self.path + 'data/test/images/'
            self.labels_path = self.path + 'data/test/labels/'

        self.filenames = os.listdir(self.images_path)

    def normalize(self, array):
        """Normalizes input arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))


    def __len__(self):
        '''Returns the total number of images'''
        return len(self.filenames)
        

    def __getitem__(self, idx):
        
        '''In case of patching returns several images, otherwise one (but we are going to use patching since the images are too big)'''

        im_name = self.filenames[idx]
        lbl_name = self.filenames[idx] + 'f'

        # Reading images

        img = (rasterio.open(self.images_path + im_name))
        lbl = (rasterio.open(self.labels_path + lbl_name))

        img = np.array([self.normalize(img.read(1)), self.normalize(img.read(2)), self.normalize(img.read(3))])
        lbl = np.array((lbl.read(1) > 0.5).astype(float))

        if self.patch_size:

            img_patches = []
            label_patches = []

            for i in range(0, img.shape[1], self.patch_size):
                for j in range(0, img.shape[2], self.patch_size):
                    if (i + self.patch_size < img.shape[1]) and (j + self.patch_size < img.shape[2]):
                        img_patch = img[:, i:i+self.patch_size, j:j+self.patch_size]
                        label_patch = lbl[i:i+self.patch_size, j:j+self.patch_size]

                        # Checking if the image acutally has some lakes and not just black
                        if 1 in label_patch and not (img_patch == 0).all():

                            
                            img_patch = ToTensor()(img_patch).permute(1,2,0)
                            label_patch = ToTensor()(label_patch)
                            img_patches.append(img_patch)
                            label_patches.append(label_patch)

            return torch.stack(img_patches, dim=0), torch.stack(label_patches)
        
        return ToTensor()(img), ToTensor()(lbl)

#blablabla