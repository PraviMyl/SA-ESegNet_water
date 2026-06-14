from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools
import os
import glob
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
import time
from skimage import io, color
import sys
import json
from skimage.color.colorconv import xyz2lab

from skimage.transform import resize


os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=fast_compile'

import keras.models as models
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras.models import Model
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


np.random.seed(1337) # for reproducibility



def normalized(rgb):

    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm


def binarylab(labels):
    x = np.zeros([labels.shape[0],labels.shape[1],2])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            a = int(labels[i][j]/255)
            x[i,j,a]=1
    return x



################# Model  #############################

# Load the Model
with open('SA-ESegNet.json') as model_file:
    model = models.model_from_json(model_file.read())


# for i, layer in enumerate(model.layers):
#     print(i, layer.name, layer.trainable)

# Freeze upto conv_11 layer
for layer in model.layers[:64]:
    layer.trainable = False

x = model.output
ft_model = Model(inputs=model.input, outputs=x)


# Make sure the frozen layers are correct
for i, layer in enumerate(ft_model.layers):
    print(i, layer.name, layer.trainable)

# load the fine-tuned weights
ft_model.load_weights("./weights/ft_model_weights.hdf5")


########################### Evaluation ###########################

# Define the patch size and stride
patch_size = (300,300)
stride = (150, 150)

def preprocess_image_into_patches(image):
    # Extract patches from the input image
    patches = []
    height, width, _ = image.shape
    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patch = normalized(cv2.resize(patch, (256,256)))
            patches.append(patch)
    return np.array(patches)

def preprocess_shadow_into_patches(image):
    # Extract patches from the input image
    patches = []
    height, width, _ = image.shape
    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patch = cv2.resize(binarylab(patch[:,:,0]), (256,256))
            patches.append(patch)
    return np.array(patches)

def reassemble_masks(masks):
    # Reassemble the predicted masks into a single output image
    # height, width = masks.shape[1:3]
    height, width, _ = image.shape
    output_mask = np.zeros((height, width), dtype=np.uint8)
    for i, mask in enumerate(masks):
        y = (i // ((width - patch_size[1]) // stride[1] + 1)) * stride[0]
        # print(y)
        x = (i % ((width - patch_size[1]) // stride[1] + 1)) * stride[1]
        # print(x)
        output_mask[y:y+patch_size[0], x:x+patch_size[1]] = mask.argmax(axis=-1)
    return output_mask

def postprocess_mask(mask):
    # Post-process the output mask as necessary (e.g. thresholding, morphology, etc.)
    threshold = 0.9
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    return dilated_mask



tf.data.experimental.enable_debug_mode()
# Load a new image to segment
image = cv2.imread('./Dataset_III/images/t22.png')
img_shad = cv2.imread('./Dataset_III/shadow/t22.png')
img_annot = cv2.imread('./Dataset_III/lables/t22.png')
# print(image.shape)
# print(img_shad.shape)

# Preprocess the image into patches
patches = preprocess_image_into_patches(image)
patches_shadow = preprocess_shadow_into_patches(img_shad)
print(patches.shape)
print(patches_shadow.shape)

# Predict the segmentation masks for each patch using the trained SegNet model
predicted_masks = ft_model.predict([patches, patches_shadow])
# print(predicted_masks.shape)


new_shape = (predicted_masks.shape[0], 300,300, 2)
predicted_mask = np.zeros(new_shape)

for i in range(predicted_masks.shape[0]):
    predicted_mask[i] = resize(predicted_masks[i], new_shape[1:], anti_aliasing=True)

# Reassemble the predicted masks into a single output image
# predicted_masks = cv2.resize(predicted_masks,(300, 300))
output_mask = reassemble_masks(predicted_mask)

# Post-process the output mask as necessary (e.g. thresholding, morphology, etc.)
output_mask_processed = postprocess_mask(output_mask)

fig, ax = plt.subplots(1, 3, figsize=(20, 20))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input Image')
ax[1].imshow(img_annot, cmap='gray')
ax[1].set_title('Actual Label')
ax[2].imshow(output_mask_processed, cmap='gray')
ax[2].set_title('Segmentation Map')
plt.show()

cv2.imwrite('./output/t22.png', output_mask_processed)