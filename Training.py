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
from PIL import  Image
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
train_path = "./Dataset_II/train/"
train_label_path = "./Dataset_II/trainannot/"

test_path = "./Dataset_II/test/"
test_label_path = "./Dataset_II/testannot/"

train_shadowlabel_path = "./Dataset_II/train_shadow/"
test_shadowlabel_path = "./Dataset_II/test_shadow/"

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

# channel attention module
def channel_attention(input_feature, ratio=8):

	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	avg_pool = GlobalAveragePooling2D()(input_feature)
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)

	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)

	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature])

# spatial attention module
def spatial_attention(input_feature):
	kernel_size = 7

	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature

	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature.shape[-1] == 1

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature])

# Residual convolutional layer
def res_conv_block_2(x, filter_size, size, dropout, batch_norm=True):
    weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    bias_initializer=tf.keras.initializers.Zeros()

    conv = Convolution2D(size, (filter_size, filter_size), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)

    attention_module = channel_attention(conv, ratio=8)

    conv = Convolution2D(size, (filter_size, filter_size), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(attention_module)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)



    shortcut = Convolution2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=3)(shortcut)

    res_path = tf.keras.layers.Add()([shortcut, conv])
    res_path = Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path





data_shape = 256*256
img_w = 256
img_h = 256
n_labels = 2
kernel=3
pool_size=2
output_mode="softmax"
FILTER_NUM = 64 # number of filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter (kernel)
UP_SAMP_SIZE = 2 # size of upsampling filters


input_shape = (256,256,3)
input_shape1 = (256,256,2)

inputs_1 = Input(shape=input_shape)
inputs_2 = Input(shape=input_shape1)



weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
bias_initializer=tf.keras.initializers.Zeros()

# CAR
conv_1 = res_conv_block_2(inputs_1, FILTER_SIZE, FILTER_NUM, dropout=0.0, batch_norm=True)


# shadow detection
conv_2s_s = Convolution2D(64, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(inputs_2)
conv_2s_s = BatchNormalization()(conv_2s_s)
conv_2s_s = Activation('relu')(conv_2s_s)

spatial_feature1 = spatial_attention(conv_2s_s)

##concatanate the feature maps
x_concat = tf.keras.layers.Concatenate(axis=-1)([conv_1, spatial_feature1])



drop1 = Dropout(0.1)(x_concat)

pool_1 = MaxPooling2D(pool_size = (pool_size,pool_size))(drop1)

conv_3 = Convolution2D(128, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(pool_1)
conv_3 = BatchNormalization()(conv_3)
conv_3 = Activation('relu')(conv_3)
conv_4 = Convolution2D(128, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_3)
conv_4 = BatchNormalization()(conv_4)
conv_4 = Activation('relu')(conv_4)


drop2 = Dropout(0.1)(conv_4)

pool_2 = MaxPooling2D(pool_size = (pool_size,pool_size))(drop2)

conv_5 = Convolution2D(256, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(pool_2)
conv_5 = BatchNormalization()(conv_5)
conv_5 = Activation('relu')(conv_5)
conv_6 = Convolution2D(256, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_5)
conv_6 = BatchNormalization()(conv_6)
conv_6 = Activation('relu')(conv_6)
conv_7 = Convolution2D(256, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_6)
conv_7 = BatchNormalization()(conv_7)
conv_7 = Activation('relu')(conv_7)

drop3 = Dropout(0.1)(conv_7)

pool_3 = MaxPooling2D(pool_size = (pool_size,pool_size))(drop3)

conv_8 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(pool_3)
conv_8 = BatchNormalization()(conv_8)
conv_8 = Activation('relu')(conv_8)
conv_9 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_8)
conv_9 = BatchNormalization()(conv_9)
conv_9 = Activation('relu')(conv_9)
conv_10 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_9)
conv_10 = BatchNormalization()(conv_10)
conv_10 = Activation('relu')(conv_10)

drop4 = Dropout(0.1)(conv_10)

pool_4 = MaxPooling2D(pool_size = (pool_size,pool_size))(drop4)

conv_11 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(pool_4)
conv_11 = BatchNormalization()(conv_11)
conv_11 = Activation('relu')(conv_11)
conv_12 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_11)
conv_12 = BatchNormalization()(conv_12)
conv_12 = Activation('relu')(conv_12)
conv_13 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_12)
conv_13 = BatchNormalization()(conv_13)
conv_13 = Activation('relu')(conv_13)

drop5 = Dropout(0.1)(conv_13)

pool_5 = MaxPooling2D(pool_size = (pool_size,pool_size))(drop5)

print("Build encoder done..........")


#decoder

unpool_1 = UpSampling2D()(pool_5)
# print(unpool_1.shape)

conv_14 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(unpool_1)
conv_14 = BatchNormalization()(conv_14)
conv_14 = Activation('relu')(conv_14)
conv_15 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_14)
conv_15 = BatchNormalization()(conv_15)
conv_15 = Activation('relu')(conv_15)
conv_16 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_15)
conv_16 = BatchNormalization()(conv_16)
conv_16 = Activation('relu')(conv_16)

drop6 = Dropout(0.2)(conv_16)

#ZeroPadding2D(((1,0), (1,0))),

unpool_2 = UpSampling2D()(drop6)
# print(unpool_2.shape)

conv_17 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(unpool_2)
conv_17 = BatchNormalization()(conv_17)
conv_17 = Activation('relu')(conv_17)
conv_18 = Convolution2D(512, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_17)
conv_18 = BatchNormalization()(conv_18)
conv_18 = Activation('relu')(conv_18)
conv_19 = Convolution2D(256, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_18)
conv_19 = BatchNormalization()(conv_19)
conv_19 = Activation('relu')(conv_19)

drop7 = Dropout(0.2)(conv_19)

#conv_19 = ZeroPadding2D(((1, 0), (0, 0)))(conv_19) #### to change the shape from (44, 60) to (45, 60)

unpool_3 = UpSampling2D()(drop7)
# print(unpool_3.shape)

conv_20 = Convolution2D(256, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(unpool_3)
conv_20 = BatchNormalization()(conv_20)
conv_20 = Activation('relu')(conv_20)
conv_21 = Convolution2D(256, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_20)
conv_21 = BatchNormalization()(conv_21)
conv_21 = Activation('relu')(conv_21)
conv_22 = Convolution2D(128, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_21)
conv_22 = BatchNormalization()(conv_22)
conv_22 = Activation('relu')(conv_22)

#ZeroPadding2D(((1, 0), (1, 0))), #### to change the shape from (44, 60) to (45, 60)

drop8 = Dropout(0.2)(conv_22)


unpool_4 = UpSampling2D()(drop8)
# print(unpool_4.shape)

conv_23 = Convolution2D(128, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(unpool_4)
conv_23 = BatchNormalization()(conv_23)
conv_23 = Activation('relu')(conv_23)
conv_24 = Convolution2D(64, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(conv_23)
conv_24 = BatchNormalization()(conv_24)
conv_24 = Activation('relu')(conv_24)

#ZeroPadding2D(((1, 0), (1, 0))), #### to change the shape from (44, 60) to (45, 60)

drop9 = Dropout(0.2)(conv_24)

unpool_5 = UpSampling2D()(drop9)



conv_25 = Convolution2D(64, (kernel, kernel), padding='same', kernel_initializer=weight_initializer,kernel_regularizer=l2(0.00005),bias_initializer=bias_initializer)(unpool_5)
conv_25 = BatchNormalization()(conv_25)
conv_25 = Activation('relu')(conv_25)
#conv_25 = Dropout(0.5)

# spatial_feature1 = spatial_attention(conv_25)

#print(conv_25.shape)

conv_26 = Convolution2D(n_labels, (1, 1), padding='valid')(conv_25)
conv_26 = BatchNormalization()(conv_26)
# conv_26 = Reshape((img_h * img_w, n_labels))(conv_26)
conv_26 = Dense(2)(conv_26)

outputs = Activation('softmax')(conv_26)
print("Build decoder done...")

model = Model(inputs=[inputs_1, inputs_2], outputs=outputs, name="SA-ESegNet")

model.summary()
with open('SA-ESegNet.json', 'w') as outfile:
      outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
print("model saved!!!!!!!!!!!")


################## Training ##########################
lr = 1e-2
epochs = 3
batch_size = 16

sgd = tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9, decay = lr/epochs)
model.compile(loss="binary_crossentropy", optimizer= sgd, metrics=[tf.keras.metrics.CategoricalAccuracy()])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
lr_plat = ReduceLROnPlateau(patience = 5, mode = 'min')

callbacks_list = [checkpoint, lr_plat]

history = model.fit(x = [X_train, X_train_shadow], y = y_train, callbacks=callbacks_list, batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data = ([X_test, X_test_shadow], y_test), shuffle=True)

# save the trained model weights
model.save_weights('./weights/pre-trained_model_weight.hdf5')


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
