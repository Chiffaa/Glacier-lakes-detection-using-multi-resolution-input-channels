import wget
import os
import zipfile
from PIL import Image
import rasterio
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

def download_dataset(data_path):
    url = 'https://github.com/Chiffaa/Glacier-lakes-detection-using-multi-resolution-input-channels/archive/refs/heads/main.zip'
    output_directory = 'lakes.zip'
    if output_directory not in os.listdir(data_path):
        lakes = wget.download(url, out=output_directory)

    with zipfile.ZipFile(output_directory) as archive:
        for file in archive.namelist():
            if file.startswith('Glacier-lakes-detection-using-multi-resolution-input-channels-main/data/'):
                archive.extract(file[67:])   
            

class LakesDataset(Dataset):

    def __init__(self, data_path=None, download=False, transforms=None, train=True):
        if download:
            download_dataset(data_path)


        # The data should be structured in the working folder as {data: {train: [images, labels], test: [images, labels]}]}

        if train:
            self.images_path = 'data/train/images/'
            self.labels_path = 'data/train/labels/'

        else: 
            self.images_path = 'data/test/images/'
            self.labels_path = 'data/test/labels/'

        self.filenames = os.listdir(self.images_path)

    def normalize(self, array):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))


    def __len__(self):
        '''Returns the total number of images'''
        return len(self.filenames)
        

    def __getitem__(self, idx):
        im_name = self.filenames[idx]
        lbl_name = self.filenames[idx] + 'f'
        print(self.images_path + im_name)
        print(self.labels_path + lbl_name)

        img = (rasterio.open(self.images_path + im_name))
        lbl = (rasterio.open(self.labels_path + lbl_name))

        img = ToTensor()(np.dstack((self.normalize(img.read(1)), self.normalize(img.read(2)), self.normalize(img.read(3)))))
        lbl = ToTensor()(self.normalize(lbl.read(1)))

        return img, lbl        