import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.metrics import BinaryAccuracy, Recall, Precision
import numpy as np

PATCH_SIZE=256
BATCH_SIZE=16
NUM_EPOCHS=100


optim = keras.optimizers.Adam(0.0001)
# ModelCheckpoint for saving weights of the best model
callbacks = [ModelCheckpoint(f"model_{str(PATCH_SIZE)}.h5", verbose=1, save_best_only=True),
              ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
              CSVLogger(f"log_{str(PATCH_SIZE)}.csv")]

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# Dice loss and focal loss are good when we have class imbalance 
dice_loss = sm.losses.DiceLoss() 
focal_loss = sm.losses.BinaryFocalLoss() 
total_loss = dice_loss + (1 * focal_loss)

# Metrics: intersection over union (IOU), F1 score
metrics = [BinaryAccuracy(), Recall(thresholds=0.5), Precision(thresholds=0.5), sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

model_dict = {'dice_loss_plus_1binary_focal_loss':total_loss, 'iou_score':sm.metrics.IOUScore(threshold=0.5), 'f1-score':sm.metrics.FScore(threshold=0.5)}

def divide_image(img, lbl, output_size):
    images = []
    labels = []
    for i in range(0, img.shape[0], output_size):
        for j in range(0, img.shape[1], output_size):
            if (i + output_size <= img.shape[0]) and (j + output_size <= img.shape[1]):
                img_patch = img[i:i+output_size, j:j+output_size, :]
                label_patch = lbl[i:i+output_size, j:j+output_size]
                images.append(img_patch)
                labels.append(label_patch)
    return np.array(images), np.array(labels)

def reconstruct_label(lbls):
    input_size = lbls.shape[1]
    num_images = lbls.shape[0]

    output_size = int(input_size * (num_images ** 0.5))

    label = np.zeros((output_size, output_size, 1))

    for i in range(len(lbls)):
        x = (i % int(num_images ** 0.5)) * input_size
        y = int(i / int(num_images ** 0.5)) * input_size
        label[y:y+input_size, x:x+input_size, :] = lbls[i]
        

    return label
    
