import os
import rasterio
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torch
from PIL import Image

class CropImages:
    def __init__(self, input_path, patch_size):
        self.path = input_path
        self.patch_size = patch_size
        self.transform = T.ToPILImage()

    def normalize(self, array):
        """Normalizes input arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))

    def save_patches(self, img_name, folder):
        img = (rasterio.open(self.path + folder + '/images/' + img_name))
        lbl = (rasterio.open(self.path + folder + '/labels/' + img_name + 'f'))

        img = np.array([self.normalize(img.read(1)), self.normalize(img.read(2)), self.normalize(img.read(3))], dtype=np.float32)
        lbl = np.array((lbl.read(1) > 0.5).astype(np.float32))

        for i in range(0, img.shape[1], self.patch_size):
            for j in range(0, img.shape[2], self.patch_size):
                if (i + self.patch_size < img.shape[1]) and (j + self.patch_size < img.shape[2]):
                    img_patch = img[:, i:i+self.patch_size, j:j+self.patch_size]
                    label_patch = lbl[i:i+self.patch_size, j:j+self.patch_size]

                    # Checking if the image acutally has some lakes and not just black
                    if 1 in label_patch and not (img_patch == 0).all():

                        # Convertin to PIL image
                        img_patch = ToTensor()(img_patch).permute(1,2,0)
                        label_patch = ToTensor()(label_patch)

                        # checking if at least 70% of the image is not black
                        if torch.sum(img_patch == 0)/(self.patch_size * self.patch_size * 3) < 0.3:

                            if not os.path.exists(self.path[:-1] + '_' + str(self.patch_size) + '/' + folder + '/images/'):
                                os.makedirs(self.path[:-1] + '_' + str(self.patch_size) + '/' + folder + '/images/')
                                os.makedirs(self.path[:-1] + '_' + str(self.patch_size) + '/' + folder + '/labels/')


                            (self.transform(img_patch)).save(self.path[:-1] + '_' + str(self.patch_size) + '/' + folder + '/images/' + img_name[:-4] + '_'+str(i) +'_' + str(j) + '.tif')
                            (self.transform(label_patch)).save(self.path[:-1] + '_' + str(self.patch_size) + '/' + folder + '/labels/' + img_name[:-4] + '_'+str(i) +'_' + str(j) + '.tiff')
                        


    def crop_images(self):
        folders = os.listdir(self.path)
        for folder in folders:
            img_names = os.listdir(self.path + folder + '/images')
            for img in img_names:
                self.save_patches(img, folder)

