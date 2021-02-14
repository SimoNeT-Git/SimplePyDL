# Lab 4. GenerativeDeepLearning

# Main parts:
#    4.1 Autoencoders (AEs)
#    4.2 Convolutional Autoencoders (CAEs)
#    4.3 Generative Adversarial Networks (GANs)

# Until now we have seen models that work with labelled data; now we move in an unsupervised setting: we have data X
# without labels, and the goal is to learn some hidden or underlying structure of the data (e.g. in ML dimensionality
# reduction, ..).
# More specifically, the goal of generative models is to take as input training samples from some distribution and learn
# model that represents that distribution; once we have that model, we can use it to generate new data.
# We want to learn Pmodel(x) similar to Pdata(x).

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image
import os

import tensorflow as tf
from keras.datasets import fashion_mnist, mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, ReLU, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras import initializers, regularizers, constraints
from keras.utils import plot_model, np_utils
from keras import backend as K
from numpy.random import randn, randint
from keras.engine import InputSpec, Layer
from scipy.stats import norm
from keras.preprocessing import image

# Note these two lines are needed to prevent an error related to running the network on the GPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
# The error is: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try
# looking to see if a warning log message was printed above.
# The previous error given was: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR


# ----------------------- 4.1 Autoencoders -----------------------

# An autoencoder is a neural network that is trained to attempt to copy its input to its output.
# The network may be viewed as consisting of two parts:
#    - an encoder function h=f(x)
#    - a decoder that produces a reconstruction r=g(h)
# Traditionally, autoencoders were used for dimensionality reduction or feature learning; recently, theoretical
# connections with latent variable models have brought autoencoders to the forefront of generative modeling.
# Autoencoders learn a “compressed representation” of input automatically by first compressing the input (encoder) and
# decompressing it back (decoder) to match the original input.
# The learning is aided by using distance function that quantifies the information loss that occurs from the lossy
# compression.

# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# convert data into float32 and normalize in range [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('Training-set shape is:', x_train.shape, 'and test-set shape is', x_test.shape)

# reshape data: reshape the images to be input to a Dense layers(e.g. (60000, 28, 28) -> (60000, 28*28))
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print('Training-set shape is', x_train.shape, 'and test-set shape is', x_test.shape)

# Build a simple autoencoder

# You can create models in Keras in two ways: using sequential API or the functional API. With the latter, you can
# declare each layer specifing the previous layer (that produces the input for the current layer) and finally build the
# model:
#     e.g.  input_layer = tf.keras.layers.Input(shape=(n,))
#           layer1 = tf.keras.layers.Dense(m)(input_layer)
#           layer2 = tf.keras.layers.Dense(m)(layer1)
#           model = tf.keras.models.Model(input_layer, layer2)
# This API allow you to create models that have a lot more flexibility (e.g. you can define models where layers connect
# to any other layers).

# size of the encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# input layer
input_img = tf.keras.layers.Input(shape=(x_train.shape[1],))
# "encoded" is the encoded representation of the input
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = tf.keras.layers.Dense(x_train.shape[1], activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = tf.keras.models.Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = tf.keras.models.Model(input_img, encoded)
# create a input layer for an encoded (32-dimensional) input to be used by decoder
encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = tf.keras.models.Model(encoded_input, decoder_layer(encoded_input))

# Compile the model
adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer=adam, loss='mse')

# Train the model
history = autoencoder.fit(x_train, x_train,
                          epochs=30,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, x_test))


# Plot training and validation loss values
def plot_hist(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


plot_hist(history)

# Visualize some outputs of the decoder r=g(h)
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Visualize some outputs of the encoder h=f(x)
encoded_imgs = encoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 8))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# ### Regularization

# Try to use a regularizer (as L1) in the encoder. How does this affect the results? What can you observe?
# Induce sparsity, less validation loss value, reconstructed images less accurate
# Hints:
#   - Dense layer has the activity_regularizer parameter https://keras.io/api/layers/regularizers/
#   - Let's train this model for more epochs (with the added regularization the model is less likely to overfit and
#   can be trained longer).
encoding_dim = 32
l1_reg = tf.keras.regularizers.l1(10e-5)
input_img = tf.keras.layers.Input(shape=(x_train.shape[1],))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu',
                                activity_regularizer=l1_reg)(input_img)
decoded = tf.keras.layers.Dense(x_train.shape[1], activation='sigmoid',
                                activity_regularizer=l1_reg)(encoded)

autoencoder = tf.keras.models.Model(input_img, decoded)
encoder = tf.keras.models.Model(input_img, encoded)
encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = tf.keras.models.Model(encoded_input, decoder_layer(encoded_input))

# Compile the model
autoencoder.compile(optimizer=adam, loss='mse')  # or loss='binary_crossentropy'

# Train the model
history = autoencoder.fit(x_train, x_train,
                          epochs=60,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, x_test))

plot_hist(history)

# Visualize some outputs of the decoder r=g(h)
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# ----------------------- 4.2 Convolutional Autoencoders -----------------------

# Convolutional Autoencoders combines Autoencoder with the convolutional operation: this operation allows Neural
# Networks to make a qualitative leap in the visual domain, due to its properties, and have made the Convolutional
# Neural Networks (CNNs) the state-of-the-art in most scenarios.
# The key difference with CNNs is that them are trained end-to-end to learn filters and combine features with the aim of
# classifying their input (supervised learning); the CAEs are trained only to learn filters able to extract features
# that can be used to reconstruct the input.

# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print('Training-set shape is:', x_train.shape, 'and test-set shape is', x_test.shape)

# Reshape the images to be input to a Convolutional layer(e.g. (60000, 28, 28)-> (60000, 28, 28, 1))
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print('Training-set shape is:', x_train.shape, 'and test-set shape is', x_test.shape)

# Build a Convolutional Autoencoder

# Basically the CAE model, instead of Dense layers, has a set of convolutional and pooling layers (encoder) and
# convolutional and upsampling layers (decoder)

# Hint: in this example, you can use 3x3 kernels and few filters (such as 16 for the first layer and 8 for the others)
# for each layer, due to the difficulty of the problem

def out_dim_kernel(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) / stride + 1

# input image
input_img = tf.keras.layers.Input(shape=(28, 28, 1))
# x = # --fill here-- # # Conv layer with ReLU activation function
conv_1 = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=(3, 3), padding='same')(input_img)  # out(28, 28, 16)
# x = # --fill here-- # # Max pooling layers with pool_size = (2,2)
pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_1)  # out(14, 14, 16)
# x = # --fill here-- # # Conv layer with ReLU activation function
conv_2 = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=(3, 3), padding='same')(pool_1)  # out(14, 14, 8)
# x = # --fill here-- # # Max pooling layers with pool_size = (2,2)
pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2)  # out(7, 7, 8)
# x = # --fill here-- # # Conv layer with ReLU activation function
conv_3 = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=(3, 3), padding='same')(pool_2)  # out(7, 7, 8)
# encoded = # --fill here-- # # Max pooling layers with pool_size = (2,2)
encoded = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv_3)  # out(4, 4, 8)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = # --fill here-- # # Conv layer with ReLU activation function
deconv_1 = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=(3, 3), padding='same')(encoded)  # out(4, 4, 8)
# x = # --fill here-- # # Upsampling layer with size=(2,2)
upsamp_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv_1)  # out(8, 8, 8)
# x = # --fill here-- # # Conv layer with ReLU activation function
deconv_2 = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=(3, 3), padding='same')(upsamp_1)  # out(8, 8, 8)
# x = # --fill here-- # # Upsampling layer with size=(2,2)
upsamp_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv_2)  # out(16, 16, 8)
# x = # --fill here-- # # Conv layer with ReLU activation function
# Note: don't put padding='same' here otherwise the kernel-map of the final layer won't have same dimension as the input
# images (28, 28) but it will be (32, 32) = (16*2, 16*2). Without same padding, default padding value will be 0 and so
# out_dim_kernel(input_size=16, kernel_size=3, padding=0, stride=1) = 14
deconv_3 = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=(3, 3))(upsamp_2)  # out(14, 14, 16)
# x = # --fill here-- # # Upsampling layer with size=(2,2)
upsamp_3 = tf.keras.layers.UpSampling2D(size=(2, 2))(deconv_3)  # out(28, 28, 16)
# decoded = # --fill here-- # # Conv layer with sigmoid activation function (output image in range [0,1])
decoded = tf.keras.layers.Conv2D(1, activation='sigmoid', kernel_size=(3, 3), padding='same')(upsamp_3)  # out(28, 28, 1)

# autoencoder = # --fill here-- #
autoencoder = tf.keras.models.Model(input_img, decoded)

# Compile the model
adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy')  # or loss='binary_crossentropy' or 'mse'

# Train the model
history = autoencoder.fit(x_train, x_train,
                          epochs=50,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, x_test))


# Plot training and validation loss values
plot_hist(history)

# Visualize some outputs of the decoder r=g(h)
decoded_imgs = autoencoder.predict(x_test)

n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# There are different interesting practical applications of autoencoders such as data denoising, anomaly detection and
# dimensionality reduction for data visualization.

# ### Image denoising with CAEs

# Create noisy samples
noise_factor = 0.65
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.) # required to put all the values in [0,1]
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Show some noisy samples
n = 10
plt.figure(figsize=(20, 2))
for i in range(1,n):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Build a Convolutional Autoencoder
input_img = tf.keras.layers.Input(shape=(28, 28, 1))
conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # out(28, 28, 32)
pool_1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv_1)  # out(14, 14, 32)
conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)  # out(14, 14, 32)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv_2)  # out(7, 7, 32)

# at this point the representation is (7, 7, 32)

deconv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  # out(7, 7, 32)
upsamp_1 = tf.keras.layers.UpSampling2D((2, 2))(deconv_1)  # out(14, 14, 32)
deconv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upsamp_1)  # out(14, 14, 32)
upsamp_2 = tf.keras.layers.UpSampling2D((2, 2))(deconv_2)  # out(28, 28, 32)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsamp_2)  # out(28, 28, 1)

autoencoder = tf.keras.models.Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer=adam, loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train_noisy, x_train,
                epochs=150,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# Visualize reconstructed (denoised) images
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Try to play with the noise_factor value: is there a threshold from which is not possible to reconstruct the image?
# How does the increase of the noise_factor value affect the number of epochs required in the training process?


# ----------------------- 4.3. Generative Adversarial Networks -----------------------

# In this part we will build a model able to generate new images using a Deep Convolutional Generative Adversarial
# Network (DCGAN).
# At https://arxiv.org/pdf/1511.06434.pdf you can find the DCGAN paper: this latter has lot of useful suggestions in
# terms of settings about the architecture, the optimizers and the hyperparameters to use.

# Load fashion-MNIST dataset

# In this part, we will use the fashion-MNIST dataset, that cointains MNIST-like fashion product images
# (https://github.com/zalandoresearch/), to train the Generator and the Discriminator; after training, the Generator
# will be able to generate images that resembles the original fashion-MNIST data.
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# np.expand_dims allow to change the shape from (60000,28,28) to (60000,28,28,1)
# X_train = #--fill here--#
# X_test = #--fill here--#
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)
print('X_train, y_train shape:', X_train.shape, y_train.shape, '\nX_test, y_test shape:', X_test.shape, y_test.shape)

# Visualize some samples
fig = plt.figure(figsize=(8, 3))
for i in range(0, 10):
    plt.subplot(2, 5, 1 + i, xticks=[], yticks=[])
    plt.imshow(np.squeeze(X_train[i]), cmap='gray')
plt.tight_layout()

# Preprocess data
num_classes = 10

# convert class vectors to binary class matrices (you can use np_utils.to_categorical function of Keras.utils)
# y_train = #--fill here--#
# y_test = #--fill here--#
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# the generator is using tanh activation in the last layer, for which we need to preprocess
# the image data into the range between -1 and 1.
X_train = np.float32(X_train)
# X_train = #--fill here--#
X_train = (X_train / 255 - 0.5) * 2
# clip in range [-1, 1]
X_train = np.clip(X_train, -1, 1)

X_test = np.float32(X_test)
# X_test = #--fill here--#
X_test = (X_test / 255 - 0.5) * 2
# clip in range [-1, 1]
X_test = np.clip(X_test, -1, 1)

print('y_train reshape:', y_train.shape)
print('y_test reshape:', y_test.shape)

# ### Define the Generator

# The Generator takes as input random noise (low dimensionality, e.g. 100) and produce an image (high dimensionality,
# e.g. 32 x 32 x 3): in order to perform the upsampling operation we can use the transposed convolution (Conv2DTranspose
# in Keras).
# The first layer of the Generator is a Dense layer that takes random noise as input (of dimension latent_dim) and
# upsample it through layers in order to obtain the size of the image (in this case 28x28x1).
# After each Conv2DTranspose layer we can add a BatchNormalization layer (that helps during training) and a LeakyRelu
# activation layer. The output layer uses tanh activation in order to output an image in [-1,1].
def gen(latent_dim):
    rn = initializers.RandomNormal(stddev=0.02)
    model = Sequential()

    # 7x7x112
    model.add(Dense(7*7*112, input_shape=(latent_dim,), kernel_initializer=rn))
    model.add(Reshape((7, 7, 112)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # 14x14x56
    # model.add(#--fill here--#) # transposed convolution (use 5x5 kernels and stride=2)
    model.add(Conv2DTranspose(56, kernel_size=5, strides=2, padding='same', kernel_initializer=rn))
    # model.add(#--fill here--#) # batch normalization
    model.add(BatchNormalization())
    # model.add(#--fill here--#) # Leaky ReLU
    model.add(LeakyReLU(0.2))

    # 28x28x1
    # model.add(#--fill here--#) # transposed convolution (use 5x5 kernels and stride=2 with tanh activation)
    model.add(Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh', kernel_initializer=rn))

    return model


latent_dim = 100
generator = gen(latent_dim)
generator.summary()

# Plot samples generated by the Generator before training

# generate points in latent space as input for the Generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape((n_samples, latent_dim))  # reshape into a batch of inputs for the network
    return x_input


# using the function predict() generate num_samples samples and plot it using plt.imshow
def plot_generated_samples(generator, num_samples, title=""):
    x_fake = generator.predict(generate_latent_points(latent_dim, num_samples))
    fig = plt.figure(figsize=(7, 7))
    for k in range(num_samples):
        plt.subplot(np.sqrt(num_samples), np.sqrt(num_samples), k + 1, xticks=[], yticks=[])
        aux = np.squeeze(x_fake[k])
        plt.imshow(((aux + 1) * 127).astype(np.uint8), cmap='gray')
    fig.suptitle(title)
    plt.show()


plot_generated_samples(generator, 16, 'Generated samples')

# ### Define the Discriminator

# The Discriminator is a Convolutional Neural Network that distinguish between real and generated samples (basically
# it is a binary classification problem)
def dis(input_shape):

    rn = initializers.RandomNormal(stddev=0.02)
    model = Sequential()

    # 14x14x56
    # model.add(#--fill here--#) # Convolutional layer (5x5 kernels and stride=2)
    model.add(Conv2D(56, input_shape=input_shape, kernel_size=5, strides=2, padding='same', kernel_initializer=rn))
    # model.add(#--fill here--#) # LeakyReLU
    model.add(LeakyReLU(0.2))

    # 7x7x112
    # model.add(#--fill here--#) # Convolutional layer (5x5 kernels and stride=2)
    model.add(Conv2D(112, kernel_size=5, strides=2, padding='same', kernel_initializer=rn))
    # model.add(#--fill here--#)# Batch normalization
    model.add(BatchNormalization())
    # model.add(#--fill here--#)# LeakyReLU
    model.add(LeakyReLU(0.2))

    # 4x4x224
    # model.add(#--fill here--#)#  Convolutional layer (5x5 kernels and stride=2)
    model.add(Conv2D(224, kernel_size=5, strides=2, padding='same', kernel_initializer=rn))
    # model.add(#--fill here--#) # LeakyReLU
    model.add(LeakyReLU(0.2))

    # Flatten layer
    # model.add(#--fill here--#)
    model.add(Flatten())
    # Output Dense layer (binary)
    # model.add(#--fill here--#)
    model.add(Dense(1, activation='sigmoid'))

    return model


input_shape = X_train.shape[1:]
discriminator = dis(input_shape)
discriminator.summary()

# Classify some generated samples before training
num_samples = 3
latend_dim = 100

generated_samples = generator.predict(generate_latent_points(latent_dim, num_samples))
print('Shape of the generated samples is', generated_samples.shape)
decision = discriminator.predict(generated_samples)
print("Discriminator's decision is\n", decision)

# ### Define the GAN model

# In order to create the GAN model in Keras you should create a Model that incorporate both the Generator and the
# Discriminator.
# In this lab we use Adam optimizer, as suggested in the DCGAN paper, but you can change it (e.g. SGD, RMSprop, etc..)
# and the hyperparameters (learning rate, momentum, etc..) to see how it affects the training phase.
def GAN(discriminator_lr, generator_lr, discriminator, generator):
    discriminator.compile(Adam(lr=discriminator_lr, beta_1=0.5), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    discriminator.trainable = False

    z = Input(shape=(latent_dim,))
    hidden = generator(z)
    output = discriminator(hidden)
    model = Model(inputs=z, outputs=output)

    model.compile(Adam(lr=generator_lr, beta_1=0.5), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


discriminator_lr=0.0002
generator_lr = 0.0002
gan = GAN(discriminator_lr, generator_lr, discriminator, generator)
gan.summary()


# Utils function to plot the losses
def plot_loss(d_loss,g_loss, dim):
    plt.close()
    plt.figure(figsize=dim)
    plt.plot(d_loss)
    plt.plot(g_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Discriminator', 'Generator'], loc='upper right')
    plt.show()


# Training

# In order to train the model, firstly set the batch_size and how many iterations the training loop will do; then you
# should complete the code following the original GAN paper https://arxiv.org/abs/1406.2661

# Note: You can use the train_on_batch function provides by Keras
# https://keras.io/api/models/model_training_apis/#trainonbatch-method
batch_size = 128
iterations = 1400 #you can play with this
k = 1  # value suggested in the GAN paper, in some cases changing this value can help during training

# Label smoothing (label from 1 to 0.9: it helps during training)
smooth = 0.1

real = np.ones(shape=(batch_size, 1))
fake = np.zeros(shape=(batch_size, 1))

d_loss = []
g_loss = []

init_time = time.time()

for i in range(0, iterations + 1):
    start_time = time.time()

    for _ in range(k):

        # Sample minibatch of m noise samples from noise prior pg(z)
        z = generate_latent_points(latent_dim, batch_size)
        X_fake = generator.predict_on_batch(z)

        # Sample minibatch of m examples from data generating distribution pdata(x)
        index = np.random.choice(X_train.shape[0], batch_size, replace=False)
        X_batch = X_train[index]

        # Update the Discriminator by ascending its stochastic gradient
        d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)
        d_loss_real = discriminator.train_on_batch(x=X_batch, y=real * (1 - smooth))

        # Discriminator loss
        d_loss_batch = d_loss_real[0] + d_loss_fake[0]

    # Sample minibatch of m noise samples from noise prior pg(z)
    z = generate_latent_points(latent_dim, batch_size)

    # Update the Generator by ascending its stochastic gradient
    g_loss_batch = gan.train_on_batch(x=z, y=real)

    d_loss.append(d_loss_batch)
    g_loss.append(g_loss_batch[0])

    if i % 200 == 0:
        plot_generated_samples(generator, 36)
        plot_loss(d_loss, g_loss, (6, 4))
        print('iteration = %d/%d, d_loss=%.3f, g_loss=%.3f' % (i + 1, iterations, d_loss[-1], g_loss[-1]), 100*' ')
        print("%s minutes per 10000 iterations ---\n\n" % ((time.time()-start_time)/60))


print("\n\n--- %s total minutes training time ---\n\n" % ((time.time()-init_time)/60))

# Plot samples and the losses after training
plot_loss(d_loss, g_loss, (8, 5))


def plot_real_samples(num_samples, title=""):
    fig = plt.figure(figsize=(7, 7))

    for k in range(num_samples):
        plt.subplot(np.sqrt(num_samples), np.sqrt(num_samples), k + 1, xticks=[], yticks=[])
        aux = np.squeeze(X_train[k])
        plt.imshow(((aux + 1) * 127).astype(np.uint8), cmap='gray')

    fig.suptitle(title)
    plt.show()


plot_real_samples(25, "Real images")
plot_generated_samples(generator, 25, "Generated (fake) images")

# ### Try with your own dataset:
# Find a dataset (possibly with lot of samples): if your data do not fit the RAM memory you can use a Data Generator
# that load data on the fly (here you can find an example
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) or alternatively the
# image_dataset_from_directory function provided by Keras (https://keras.io/api/preprocessing/image/)
# Preprocess your data: convert each image to have the same dimension in range [-1,1] (the last layer of the Generator
# is tanh activation layer)
# Create the GAN model: build a Generator and a Discriminator in a mirror way (~ same number of convolutional layers)
# Train your model: if something goes wrong, such as the losses saturates to zero or to big values and the Generator
# generates "garbage" images you can try to
#     - adjust the learning rate of the Generator and of the Discriminator
#     - change the architectures of the models (deeper with more complex problem)
#     - change the hyperparameters k (more step of the Discriminator per each Generator step or the opposite)
