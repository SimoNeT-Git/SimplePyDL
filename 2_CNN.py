# Lab 2: Convolutional Neural Network (CNN)

# Main parts:
#    2.1. Dataset pre-processing
#    2.2 Training a model from scratch
#    2.3 Data Augmentation
#    2.4 Using a pre-trained model
#    2.5 Comparing the models

import time
import tensorflow as tf

# Note these two lines are needed to prevent an error related to running the network on the GPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
# The error is: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try
# looking to see if a warning log message was printed above.
# The previous error given was: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR

# ----------------------- 2.1. Dataset pre-processing -----------------------

# The first thing that we need to do when we are dealing with a new dataset is to operate some pre-processing.
# Data preprocessing usually refers to the steps applied to make data more suitable for learning.
# In this section we are going to deal with:
#     2.1.1 Dataset loading
#     2.1.2 Normalization
#     2.1.3 Standardization
#     2.1.4 Splitting and label pre-processing
#     2.1.5 Label pre-processing

# ### 2.1.1 Dataset loading

# In this section we load the augmented dataset generated in the previous section
# Here we are importing the train and test set separated
(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.cifar10.load_data()
print('Training-set shape is', train_X.shape, 'and test-set shape is', test_X.shape)

# ### 2.1.2 Normalization

# One common practice in training a Neural Network is to normalize the images by dividing each pixel value by the
# maximum value that we can have, i.e. 255. The purpose of this is to obtain a mean close to 0.
# Normalizing the data generally speeds up learning and leads to faster convergence.
import numpy as np
print("Normalizing training set..")
train_X = np.asarray(train_X, dtype=np.float32) / 255  # Normalizing training set
print("Normalizing test set..")
test_X = np.asarray(test_X, dtype=np.float32) / 255  # Normalizing test set

# ### 2.1.3 Standardization

# Another common practice in data pre-processing is standardization.
# The idea about standardization is to compute your dataset mean and standard deviation in order to subtract from every
# data point x the dataset mean μ and then divide by the standard deviation σ. That is to apply the following operation:
# z = (x - μ) / σ
# The outcome of this operation is to obtain a distribution with mean equal to 0 and a standard deviation equal to 1.
# By applying normalization to our data we are making the features more similar to each other and this usually makes the
# learning process easier.


# Standardizing the data
def compute_mean_and_std(X):
    mean = np.mean(X, axis=(0, 1, 2))
    std = np.std(X, axis=(0, 1, 2))
    return [mean, std]


# For every image we subtract to it the dataset mean and we divide by the dataset standard deviation. Note: mean and std
# are computed on the training set and used for the test set as well. This is because we assume (in general) not to know
# the test set but the training set only, and also because we must fairly treat the train and test sets during the pre-
# processing (in fact if we divide every subset by their own std we won't treat them uniformly).
dataset_mean, dataset_std = compute_mean_and_std(train_X)
print("Standardizing training set..")
train_X = (train_X-dataset_mean)/dataset_std  # Standardizing the training set
print("Standardizing test set..")
test_X = (test_X-dataset_mean)/dataset_std  # Standardizing the test set

# ### 2.1.4 Splitting

# Now we just need to split our training set in order to get the validation set and convert our labels to one-hot
# representation.
from sklearn.model_selection import train_test_split
print("Splitting training set to create validation set..")
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
print('Training-set shape is', train_X.shape, 'test-set shape is', test_X.shape,
      'and validation-set shape is', valid_X.shape)

# ### 2.1.5 Label pre-processing

# Converting labels to one-hot representation. One-hot encoding is useful for categorical data. As an example to
# understand such encoding technique lets consider a “color” variable with the values “red“, “green” and “blue“.
# There are 3 categories and therefore 3 binary variables are needed: a “1” value is placed in the binary variable for
# the color and “0” values for the other colors. Therefore we will have:
#        red, green, blue
#  1)     1,    0,    0
#  2)     0,    1,    0
#  3)     0,    0,    1
from keras.utils.np_utils import to_categorical
train_Y_one_hot = to_categorical(train_Y)  # Converting training labels to one-hot representation
valid_Y_one_hot = to_categorical(valid_Y)  # Converting validation labels to one-hot representation
test_Y_one_hot = to_categorical(test_Y)    # Converting test labels to one-hot representation

# ----------------------- 2.2 Training a model from scratch -----------------------

# Now that we have properly pre-processed our data, we are going to create a convolutional model in Keras. Usually a
# convolutional model is made by two subsequent parts:
#    - A convolutional part
#    - A fully connected part

# Usually the convolutional part is made by some layers composed by
#    - convolutional layer: performs a spatial convolution over images (see Conv2D)
#    - pooling layer: used to reduce the output spatial dimension from n to 1 by averaging the n different value or
#    considering the maximum between them (see MaxPool2D)
#    - dropout layer: applied to a layer, consists of randomly "dropping out" (i.e. set to zero) a number of output
#    features of the layer during training (see DropOut)

# The convolutional part produces its output and the fully connected part ties together the received information
# in order to solve the classification problem

# Creating the model from scratch
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import accuracy_score

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Network parameters
batch_size = 64  # Setting the batch size
epochs = 6  # Setting the number of epochs
num_classes = len(categories)  # Getting the amount of classes

# Build here your keras model (try to use one or more convolutional layer, joint with pooling layer and dropout layer).
# Note: larger kernels "see" more things, but they also introduce more parameters to learn. For example, using 2
# successive 3x3 kernels generates less parameters than using one 5x5 kernel. Remember this!

start1 = time.time()

# Create the model
scratch_model = Sequential()
# 1) Add first convolutional part (conv + non_lin + pool)
scratch_model.add(Conv2D(32, activation='linear', input_shape=train_X.shape[1:],
                         # after the first layer, you don't need to specify the size of the input anymore
                         kernel_size=(3, 3), padding='same'))
# Note: 'same' padding means the value for padding is chosen such that the size of output feature-maps are the same as
# the input feature-maps. Thus, because output_size = (input_size - kernel_size + 2 * padding) / stride + 1
# assuming stride = 1 we will need a padding = (kernel_size - 1) / 2
scratch_model.add(LeakyReLU(alpha=0.1))
scratch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
scratch_model.add(Dropout(rate=0.3))  # try padding='same'
# 2) Add second convolutional part (conv + non_lin + pool)
scratch_model.add(Conv2D(32, activation='linear',
                         kernel_size=(3, 3), padding='same'))
scratch_model.add(LeakyReLU(alpha=0.1))
scratch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
scratch_model.add(Dropout(rate=0.3))  # try padding='same'
# 3) Add the readout (fully-connected) part
scratch_model.add(Flatten())
scratch_model.add(Dense(32, activation='linear'))
scratch_model.add(LeakyReLU(alpha=0.1))
scratch_model.add(Dense(num_classes, activation='softmax'))

# Compile the model with the Adam optimizer
scratch_model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Visualize the model through the summary function
print('\n', scratch_model.summary())

# Let's train the model!
scratch_model_history = scratch_model.fit(train_X, train_Y_one_hot, batch_size=batch_size, shuffle=True, epochs=epochs,
                                          validation_data=(valid_X, valid_Y_one_hot))

# Getting the results
scratch_model_train_acc = scratch_model_history.history['accuracy']
scratch_model_valid_acc = scratch_model_history.history['val_accuracy']
scratch_model_train_loss = scratch_model_history.history['loss']
scratch_model_valid_loss = scratch_model_history.history['val_loss']
print('Validation accuracy: ', scratch_model_valid_acc[-1])
# Testing the model
print("Test accuracy: ", accuracy_score(np.argmax(scratch_model.predict(test_X), axis=-1), test_Y))  # Testing the model
# Oss.: scratch_model.predict_classes(test_X) (as first argument of accuracy_score()) is deprecated

end1 = time.time()
print('Creating and training the first CNN took', (end1 - start1) / 60, 'minutes.')

# Is the obtained value coherent with what you expected?
# What are the differences when using a different batch size? Why?


# # ----------------------- 2.3 Data Augmentation -----------------------
#
# # Before even starting to load the dataset we should ask ourselves whether the available amount of data is sufficient to
# # our purposes. When the answer is negative we could need to do "data augmentation".
# # Doing data augmentation means to increase the number of available data points. In terms of images, it may mean that
# # increasing the number of images in the dataset. A common way to do this is to generate new images by applying a linear
# # transformation to the original images in the dataset.
# # The most common linear transformations are the following:
# #    - Rotation
# #    - Shifting
# #    - Blurring
# #    - Change lighting conditions
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Augmentation parameters
# noise_range = 5  # Gaussian blur range
# flip_hor_prob = 0.5	 # Probability to flip horizontally the image
# rot_range = 30  # Rotation range
#
# print('Training-set shape is', train_X.shape, 'and test-set shape is', test_X.shape)
#
# # Try different augmentation strategies
# cifar10_datagen = ImageDataGenerator(
#     featurewise_center=False,
#     featurewise_std_normalization=False,
#     # rotation_range=20,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     horizontal_flip=True)
#
# # # To visualize transformed data:
# # trans_data_iter = cifar10_datagen.flow(train_X, to_categorical(train_Y))
# # image = next(trans_data_iter)
# # import matplotlib.pyplot as plt
# # plt.imshow(image[0][0])
# # plt.show()
#
# start2 = time.time()
#
# # Build here your keras model. Try to use one or more convolutional layer, joint with pooling layer and dropout layer.
# # Create the model
# scratch_model = Sequential()
# # 1) Add first convolutional part (conv + non_lin + pool)
# scratch_model.add(Conv2D(32, activation='linear', input_shape=train_X.shape[1:],
#                          # after the first layer, you don't need to specify the size of the input anymore
#                          kernel_size=(3, 3), padding='same'))  # try padding='same'
# scratch_model.add(LeakyReLU(alpha=0.1))
# scratch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # try padding='same'
# scratch_model.add(Dropout(rate=0.3))  # try padding='same'
# # 2) Add second convolutional part (conv + non_lin + pool)
# scratch_model.add(Conv2D(32, activation='linear',
#                          kernel_size=(3, 3), padding='same'))  # try padding='same'
# scratch_model.add(LeakyReLU(alpha=0.1))
# scratch_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # try padding='same'
# scratch_model.add(Dropout(rate=0.3))  # try padding='same'
# # 3) Add the readout (fully-connected) part
# scratch_model.add(Flatten())
# scratch_model.add(Dense(32, activation='linear'))
# scratch_model.add(LeakyReLU(alpha=0.1))
# scratch_model.add(Dense(num_classes, activation='softmax'))
#
# # Compile the model with the Adam optimizer
# scratch_model.compile(
#     loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#
# # Visualize the model through the summary function
# print('\n', scratch_model.summary())
#
# # Let's train the model! Fit the model on batches with real-time data augmentation.
# scratch_model_history = scratch_model.fit(cifar10_datagen.flow(train_X, to_categorical(train_Y), batch_size=batch_size),
#                                           epochs=epochs)
#
# # Getting the results
# scratch_model_train_acc = scratch_model_history.history['accuracy']
# scratch_model_valid_acc = scratch_model_history.history['val_accuracy']
# scratch_model_train_loss = scratch_model_history.history['loss']
# scratch_model_valid_loss = scratch_model_history.history['val_loss']
# print('Validation accuracy: ', scratch_model_valid_acc[-1])
# # Testing the model
# print("Test accuracy: ", accuracy_score(np.argmax(scratch_model.predict(test_X), axis=-1), test_Y))  # Testing the model
#
# end2 = time.time()
# print('Creating and training the second CNN (with data augmentation) took', (end2 - start2) / 60, 'minutes.')
#
# # What is the performance obtained on this new augmented dataset?
# # How can you explain the obtained result?


# ----------------------- 2.4 Using a pre-trained model -----------------------

# A common alternative to train a model from scratch consists in using a pre-trained model.
# The idea is to replace the convolutional part with a highly optimized convolutional part engineered and trained
# previously by someone else.
# Usually the models that we can use through keras.applications have been trained over the image net dataset.

# After the convolutional part replacement we still need to set up a fully connected part.
# Why in this lab we cannot use the fully connected part of VGG19 Net?
# What should we do to use it?
# And more in general in which situations we can do that?
# Moreover, using a pre-trained network is not always the best choice.
# Can you guess in which situations could be useful to use a pre-trained model?

# Creating the model based over the pretrained Xception network
from keras import applications
vgg19 = applications.VGG19(weights="imagenet", include_top=False, input_shape=train_X.shape[1:])

# Produce the features of the train and validation sets using VGG19 predict function
train_X_feature = vgg19.predict(train_X)  # Producing the train feature
valid_X_feature = vgg19.predict(valid_X)  # Producing the test feature

from keras import models
from keras import layers
from keras import optimizers

start3 = time.time()

# Creating a simple model that will classify the extracted features from the VGG19 network
pretrained_model = models.Sequential()
pretrained_model.add(layers.Flatten())
pretrained_model.add(layers.Dense(64, activation='linear'))
pretrained_model.add(layers.LeakyReLU(alpha=0.1))
pretrained_model.add(layers.Dropout(rate=0.3))
pretrained_model.add(layers.Dense(num_classes, activation='softmax'))
pretrained_model.compile(optimizer=optimizers.RMSprop(lr=2e-4), loss='categorical_crossentropy', metrics=['acc'])

# Visualize the model through the summary function
print('\n', vgg19.summary())

# Let's train the model!
pretrained_model_history = pretrained_model.fit(train_X_feature, train_Y_one_hot, epochs=epochs, batch_size=batch_size,
                                                validation_data=(valid_X_feature, valid_Y_one_hot))

# Getting the results
pretrained_model_train_acc = pretrained_model_history.history['acc']
pretrained_model_valid_acc = pretrained_model_history.history['val_acc']
pretrained_model_train_loss = pretrained_model_history.history['loss']
pretrained_model_valid_loss = pretrained_model_history.history['val_loss']
test_X_feature = vgg19.predict(test_X)  # Producing the test feature
print('Validation accuracy: ', pretrained_model_valid_acc[-1])
# Testing the model
print("Test accuracy: ", accuracy_score(np.argmax(pretrained_model.predict(test_X_feature), axis=-1), test_Y))

end3 = time.time()
print('Creating and training the last (pre-trained) CNN took', (end3 - start3) / 60, 'minutes.')

# ----------------------- 2.5 Comparing the models -----------------------

# Now that we trained both the "from scratch" and the "pre-trained" models, we are going to compare the obtained results
# obtained during the training. We are going to consider accuracy and loss.
# What can you expect from these plots?

# Create here the plots to compare the "from scratch" model and the "pretrained" model
# Try to produce a comparison plot about the accuracies (train and validation) and another plot for the losses
# Creating the plots to compare the "from scratch" model and the "pretrained" model.

# Producing accuracy over epochs plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 7))
plt.plot(scratch_model_train_acc, label="Scratch Train Acc.", color="#4db8ff")
plt.plot(scratch_model_valid_acc, label="Scratch Valid. Acc.", color="#006bb3")
plt.plot(pretrained_model_train_acc, label="Pretrained Train Acc.", color="#ff4d4d")
plt.plot(pretrained_model_valid_acc, label="Pretrained Valid. Acc.", color="#b30000")
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend(loc='lower right', fancybox=True, shadow=True, ncol=4)
plt.grid()
plt.savefig('CNN_acc_epochs.png', dpi=300)

# Producing loss over epochs plot
fig = plt.figure(figsize=(16, 7))
plt.plot(scratch_model_train_loss, label="Scratch Train Loss", color="#4db8ff")
plt.plot(scratch_model_valid_loss, label="Scratch Valid. Loss", color="#006bb3")
plt.plot(pretrained_model_train_loss, label="Pretrained Train Loss", color="#ff4d4d")
plt.plot(pretrained_model_valid_loss, label="Pretrained Valid. Loss", color="#b30000")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=4)
plt.grid()
plt.savefig('CNN_loss_epochs.png', dpi=300)

# What information can you get from these plots?
# Are they showing what you expected?
