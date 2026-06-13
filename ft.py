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

# Data loader
train_path = "./Dataset_I/train/"
train_label_path = "./Dataset_I/trainannot/"

test_path = "./Dataset_I/test/"
test_label_path = "./Dataset_I/testannot/"

train_shadowlabel_path = "./Dataset_I/train_shadow/"
test_shadowlabel_path = "./Dataset_I/test_shadow/"

train_list = sorted(glob(train_path + "*.png"))
train_label_list = sorted(glob(train_label_path + "*.png"))

test_list = sorted(glob(test_path + "*.png"))
test_label_list = sorted(glob(test_label_path + "*.png"))

train_shadowlabel_list = sorted(glob(train_shadowlabel_path + "*.png"))
test_shadowlabel_list = sorted(glob(test_shadowlabel_path + "*.png"))


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

def make_dataset(image_list, mask_list, shadow_list):
  images = []
  masks = []
  shadows = []

  for img, mask, shadow in zip(image_list, mask_list, shadow_list):
    images.append(normalized(cv2.resize(cv2.imread(img), (256,256))))
    masks.append(binarylab(cv2.imread(mask)[:,:,0]))
    shadows.append(binarylab(cv2.imread(shadow)[:, :, 0]))


  images = np.array(images)
  masks = np.array(masks)
  shadows = np.array(shadows)

  return images, masks, shadows

X_train, y_train, X_train_shadow = make_dataset(train_list, train_label_list, train_shadowlabel_list)
X_test, y_test, X_test_shadow = make_dataset(test_list, test_label_list, test_shadowlabel_list)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train_shadow.shape)
print(X_test_shadow.shape)

################# Model  #############################

# Load the Model
with open('SA-ESegNet.json') as model_file:
    model = models.model_from_json(model_file.read())

# load the pre-trained weights
model.load_weights("./weights/pre-trained_model_weight.hdf5")

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


# Training
lr = 1e-2
epochs = 50
batch_size = 16

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9, decay = lr/epochs)
ft_model.compile(loss="binary_crossentropy", optimizer= sgd, metrics=[tf.keras.metrics.CategoricalAccuracy()])

start = time.time()
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
# early_stopping = EarlyStopping(monitor='val_loss', verbose = 1, patience=10, min_delta = .00075)

lr_plat = ReduceLROnPlateau(patience = 5, mode = 'min')

callbacks_list = [checkpoint, lr_plat]


print("######################################")
# Fit the model
history = ft_model.fit(x = [X_train, X_train_shadow], y = y_train, callbacks=callbacks_list, batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data = ([X_test, X_test_shadow], y_test), shuffle=True)

stop = time.time()
print(f"Training time: {stop - start}s")


# This save the trained model weights to this file with number of epochs
ft_model.save_weights('./weights/ft_model_weights.hdf5')


#plt.figure(1,2,1)
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#plt.figure(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


