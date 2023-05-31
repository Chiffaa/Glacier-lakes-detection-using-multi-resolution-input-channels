import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from patch_images import CropImages
from glob import glob
import rasterio
import os

class LakesDataset:
    def __init__(self, patch_size, batch_size, path='', split=0.2):
        # path to the original folder that contains data, data_256, data_512, etc.
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.path = path
        self.split=split
        (self.train_x, self.train_y),(self.val_x, self.val_y), (self.test_x, self.test_y) = self.load_dataset()

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load_dataset(self):
        
        path_or = 'data/' # Path to original 35 images and labels

        if not os.path.isdir(self.path):
            im_crop = CropImages(path_or, self.patch_size)
            im_crop.crop_images()

        images = sorted(glob(os.path.join(self.path, "images", "*")))
        labels = sorted(glob(os.path.join(self.path, "labels", "*")))

        split_size = int(self.split * len(images))
        
        train_x, val_x = train_test_split(images, test_size=split_size, random_state=42)
        train_y, val_y = train_test_split(labels, test_size=split_size, random_state=42)

        train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

        return (train_x, train_y),(val_x, val_y), (test_x, test_y)

    def read_image(self, path):
        path = path.decode()
        x = rasterio.open(path) 
        x = np.array([(x.read(1))/255, (x.read(2))/255, (x.read(3))/255], dtype=np.float32).transpose(1,2,0)
        return x

    def read_mask(self, path):
        path = path.decode()
        x = rasterio.open(path)
        x = np.expand_dims(np.array((x.read(1) > 0.5).astype(np.float32), dtype=np.float32), axis=-1)
        return x

    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self.read_image(x)
            y = self.read_mask(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([self.patch_size, self.patch_size, 3])
        y.set_shape([self.patch_size, self.patch_size, 1])
        return x, y

    def tf_dataset(self, X, Y, batch=2):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(10)
        return dataset
    
    def train_ds(self):
        return self.tf_dataset(self.train_x, self.train_y, batch=self.batch_size)
    
    def val_ds(self):
        return self.tf_dataset(self.val_x, self.val_y, batch=self.batch_size)
    
    def test_ds(self):
        return self.tf_dataset(self.test_x, self.test_y, batch=self.batch_size)


if __name__ == "__main__":

    dataset = LakesDataset(256, 16, path='data_256/')

    train_dataset = dataset.train_ds()
    val_dataset = dataset.val_ds()
    test_dataset = dataset.test_ds()

    # (train_x, train_y),(val_x, val_y), (test_x, test_y) = load_dataset(f'data_{str(PATCH_SIZE)}/')

    # train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
    # val_dataset = tf_dataset(val_x, val_y, batch=BATCH_SIZE)
    # test_dataset = tf_dataset(test_x, test_y, batch=BATCH_SIZE)

    print(f"Train: {len(dataset.train_x)}, \nValidation: {len(dataset.val_x)}, \nTest: {len(dataset.test_x)}")
