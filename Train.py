import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time
from skimage.io import imread, imshow, concatenate_images
from keras.models import Model, load_model
from keras.layers.core import Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import models
from tensorflow.keras.utils import plot_model
from keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
from Model import myUNet_ResLearning
from Loss import loss_function
assert len(tf.config.list_physical_devices('GPU')) > 0



drr_train = 'D:\Models\Data generator'


X_train = imread(drr_train+"/Low_train.tif").astype(np.float32)
Y_train = imread(drr_train+"/GT_train.tif").astype(np.float32)
X_valid = imread(drr_train+"/Low_test.tif").astype(np.float32)
Y_valid = imread(drr_train+"/GT_test.tif").astype(np.float32)

X_train = X_train/X_train.max()
Y_train = Y_train/Y_train.max()
X_valid = X_valid/X_valid.max()
Y_valid = Y_valid/Y_valid.max()

im_height = X_train.shape[1]
im_width = X_train.shape[2]

im_height_test = X_valid.shape[1]
im_width_test = X_valid.shape[2]

X_train = X_train.reshape(-1,im_height,im_width,1)
Y_train = Y_train.reshape(-1,im_height,im_width,1)
X_valid = X_valid.reshape(-1,im_height,im_width,1)
Y_valid = Y_valid.reshape(-1,im_height,im_width,1)

print('The train stack shape is:',X_train.shape)




ix = random.randint(0, len(X_train))
fig = plt.figure(figsize=(10,10))
fig.add_subplot(1,2, 1)
cmap=plt.get_cmap('magma')
plt.imshow(X_train[ix].squeeze(),cmap)
plt.axis('off')

fig.add_subplot(1,2, 2)
cmap=plt.get_cmap('magma')
plt.imshow(Y_train[ix].squeeze(),cmap)
plt.axis('off')
plt.show()





checkpoint_filepath = drr_train+'/model-Unet-Fast STED-Tubulin-3D-20210121.h5'
callbacks = [
    EarlyStopping(patience=200, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1, min_lr=1e-6),
    ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True)
]
mm = myUNet_ResLearning(filters=[64,128,256,512,1024])
opt = Adam(learning_rate=5e-5)
mm.compile(optimizer=Adam(),loss=loss_function(0,0,0,1).percep_loss, metrics=['mse'])
results = mm.fit(X_train[0:20], Y_train[0:20],batch_size=16, epochs=1,validation_data=(X_valid, Y_valid))

