import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Cropping2D, BatchNormalization, Activation, Conv2DTranspose
from keras.models import Model
from model import build_unet

# Pyrimid approach

def pyramid_unet(input_shape, num_scales):
    input_layers = []
    output_layers = []

    input = Input(input_shape)

    # Create input and output layers for each scale
    for scale in range(num_scales):
        scale_factor = 2 ** scale
        scaled_shape = (input_shape[0] // scale_factor, input_shape[1] // scale_factor, input_shape[2])
        #print(scaled_shape)
        input_layer = tf.image.resize(input,[input_shape[0] // scale_factor, input_shape[1] // scale_factor])
        model = build_unet(scaled_shape, name="UNET_" + str(scale))
        output_layer = model(input_layer)

        input_layers.append(input_layer)
        output_layers.append(output_layer)

    for scale in range(num_scales):
        i = 0
        while i < scale:
            output_layers[scale] = UpSampling2D(size=(2, 2))(output_layers[scale])
            i += 1


    merged = Concatenate()(output_layers)

    # Final convolutional layer
    output = Conv2D(1, 1, activation='sigmoid')(merged)
    #print(tf.convert_to_tensor(input_layers))

    model = Model(inputs=input, outputs=output)
    return model


if __name__=="__main__":
    
    image = tf.random.uniform((1, 256, 256, 3))
    p_unet = pyramid_unet((256, 256, 3), 3)
    p_res = p_unet(inputs=image)

    print(p_res.shape)