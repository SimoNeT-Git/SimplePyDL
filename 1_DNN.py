# Lab1: Deep Neural Network (DNN)

# Main parts:
#    1.1. Keras Basics
#    1.2. Building Deep Neural Networks
#    1.3. Overfitting

# ----------------------- 1.1. Keras Basics -----------------------

# In this section we will:
#     1.1.1 Build a Single Layer Perceptron (SLP)
#     1.1.2 Build a Multi-Layer Perceptron (MLP)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# ### 1.1.1 Build a Single Layer Perceptron (SLP)

# ## Using tensorflow
def one_dense_layer(x, n_in, n_out):  # n_in: number of inputs, n_out: number of outputs
    # W = [1,1]
    W = tf.ones((n_in, n_out))
    # b = 1
    b = tf.ones((1, n_out))
    # z = W*x + b
    z = tf.matmul(x, W) + b
    # y = sigmoid(z)
    out = tf.sigmoid(z)
    return out


x_input = tf.constant([[1, 2.]], shape=(1, 2))
res = one_dense_layer(x_input, n_in=2, n_out=2)
# print('Output of a very simple SLP with tensorflow:', res)

# ## Using keras
# Define the number of inputs and outputs
n_input_nodes = 2
n_output_nodes = 2
# First define the model
model = Sequential()  # lets define a linear stack of network layers
# define our single fully connected network layer
dense_layer = Dense(n_output_nodes, activation='sigmoid', kernel_initializer="Ones", bias_initializer="Ones")
# Add the dense layer to the model
model.add(dense_layer)

# ### 1.1.2 Build a Multi-Layer Perceptron (MLP)

# MLPs are fully connected, each node in one layer connects with a certain weight to every node in the following layer.
# Try to build one composed by two hidden dense layer with ReLU activation and one dense output layer(units=1) with
# sigmoid activation.

# Generate dummy data with N = 100 features
N = 100
n_train, n_test = 1000, 100
train_data = np.random.random((n_train, N))
train_labels = np.random.randint(2, size=(n_train, 1))
test_data = np.random.random((n_test, N))
test_labels = np.random.randint(2, size=(n_test, 1))

# ## Build a simple MLP
# Create a Sequential object and then add 3 Dense layers
units = 32
# Create a Sequential
model = Sequential()
# Add a Dense layer with 32 neurons, with relu as activation function and input dimension
# equal to the number of features
model.add(Dense(units, activation='relu', input_dim=N))
# Add a Dense layer with 32 neurons, with relu as activation function
model.add(Dense(units, activation='relu'))
# To produce the output Add a Dense layer with 1 neurons, with sigmoid as activation function
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
# The fit function output is a History object. The history.history attribute is a record of
# training loss values and metrics values at successive epochs, as well as validation loss values
# and validation metrics values
history = model.fit(train_data, train_labels, epochs=30, batch_size=128)
_, train_acc = model.evaluate(train_data, train_labels, verbose=1)
_, test_acc = model.evaluate(test_data, test_labels, verbose=1)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))

# ----------------------- 1.2. Building Deep Neural Networks -----------------------

# In this section we will:
#     1.2.1 Import the dataset
#     1.2.2 Build a model
#     1.2.3 Train the model
#     1.2.4 Evaluate the model

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, RMSprop

# ### 1.2.1 Import the dataset

# Fashion-MNIST is a dataset of Zalando’s article images consisting of a training set of 60,000 examples and a test set
# of 10,000 examples. Each example is a 28×28 grayscale image ([0 - 255]), associated with a label from 10 classes.
mnist_fashion = tf.keras.datasets.fashion_mnist
(x_learn, y_learn), (x_test, y_test) = mnist_fashion.load_data()
# Normalize data
x_learn, x_test = x_learn / 255.0, x_test / 255.0
# Split learning set into training set and validation set (sklearn provide train_test_split() method)
x_train, x_val, y_train, y_val = train_test_split(x_learn, y_learn, test_size=0.3)
num_classes = 10  # Fashion-MNIST classes
print('Training set has shape', x_train.shape, 'Validation set has shape', x_val.shape,
      'and Test set has shape', x_test.shape)

# Plot some samples from the training set
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

# ### 1.2.2 Build a Model

# A Deep Neural Network is a neural network composed by many layers and consequently it has a deeper structure.
# The number of layers in the network depends on different factors: for example on the data available, on the complexity
# of the problem, on the computational power and so on. The value produced as output by a neuron is determined by
# the input the neuron receives and by the activation function. There exists different choices for the activation
# function. One of the most used is Relu but it depends on the data and on the network architecture. Other possibilities
# are: Sigmoid, Tanh, Leaky ReLu, Maxout and ELU.

# Build a model with this structure: Flatten + 4 x Dense(ReLU) + Dense(softmax)
# https://keras.io/layers/core/
model = Sequential()
model.add(Flatten())
# Add a Dense layer with 512 neurons, with relu as activation function
model.add(Dense(512, activation='relu'))
# Add a Dense layer with 256 neurons, with relu as activation function
model.add(Dense(526, activation='relu'))
# Add a Dense layer with 128 neurons, with relu as activation function
model.add(Dense(128, activation='relu'))
# Add a Dense layer with 64 neurons, with relu as activation function
model.add(Dense(64, activation='relu'))
# Add a Dense layer with number of neurons equal to the number of classes, with softmax as activation function
model.add(Dense(num_classes, activation='softmax'))

# When we are building a model there are many design choises that we must operate: Loss Function, Metrics and Optimizer.
# - Loss functions are used to compare the network's predicted output with the real output, in each pass of the
# backpropagations algorithm; common loss functions are: mean-squared error, cross-entropy, and so on...
# - Metrics are used to evaluate a model; common metrics are precision, recall, accuracy, auc,..
# - The Optimizer determines the update rules of the weights. The performance and update speed may heavily vary from
# optimizer to optimizer; in choosing an optimizer what's important to consider is the network depth, the type of layers
# and the type of data. Common optimizers are: sgd, adagrad, momentum, nag, adadelta, rmsprop.

# Optimizers:    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
adad = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adag = Adagrad(lr=0.01, epsilon=None, decay=0.0)
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Losses:    https://keras.io/losses/
loss = ['sparse_categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error',
        'categorical_crossentropy', 'categorical_hinge']

# Metrics:    https://www.tensorflow.org/api_docs/python/tf/metrics
metrics = ['accuracy', 'precision', 'recall']

# Compile the model you created before. Use:
# -> adam optimizer as optimizer
# -> sparse categorical crossentropy as loss function
# -> accuracy as metric
model.compile(optimizer=adam, loss=loss[0], metrics=[metrics[0]])

# ### 1.2.3 Train the model

# The batch size is a number of samples processed before the model is updated.
# The number of epochs is the number of complete passes through the training dataset.
batch_size = 200
epochs = 50

# Fit your model and save the returned value as "history". Set properly both the batch size value and the epochs value.
# Use both the train and validation set.
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs)


# History visualization
def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


plot_history(history)

# What could you notice in the loss graph training the model over large number of epochs?
# Training loss continue to decrease in a flatten way until it goes near 0; validation loss after a while starts to
# increase significantly -> OVERFITTING

# ### 1.2.4 Evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=1)
_, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))

# Try to play with these parameters (loss and optimizers) in order to see how this choice affects the accuracy.
# What do you expect? Which is faster?


# ----------------------- 1.3. Overfitting -----------------------

# Given some training data and a network architecture, there are multiple sets of weights values (multiple models) that
# could explain the data, and simpler models are less likely to overfit than complex ones.
# A "simple model" in this context is a model where the distribution of parameter values has less entropy (or a model
# with fewer parameters altogether).
# How to improve generalization of our model on unseen data?

# In this section we will:
#     1.3.1 Add weight regularization
#     1.3.2 Dropout
#     1.3.3 Early stopping

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ### 1.3.1 Add weight regularization

# A common way to mitigate overfitting is to put constraints on the complexity of a network by forcing its weights only
# to take small values, which makes the distribution of weight values more "regular".
# This is called "weight regularization", and it is done by adding to the loss function of the network a cost associated
# with having large weights.
# This cost comes in two flavors:
#   -  L1 regularization
#   -  L2 regularization
# In tf.keras, weight regularization is added by passing weight regularizer instances to layers as keyword arguments.

# Build the model
model = Sequential()
model.add(Flatten())
# Add a Dense layer with 512 neurons, with relu as activation function and a l2 kernel regularizer (with 0.001 as
# regularization value)
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# Add a Dense layer with 256 neurons, with relu as activation function and a l2 kernel regularizer (with 0.001 as
# regularization value)
model.add(Dense(526, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# Add a Dense layer with 128 neurons, with relu as activation function and a l2 kernel regularizer (with 0.001 as
# regularization value)
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# Add a Dense layer with 64 neurons, with relu as activation function and a l2 kernel regularizer (with 0.001 as
# regularization value)
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# Add a Dense layer with number of neurons equal to the number of classes, with softmax as activation function
model.add(Dense(num_classes, activation='softmax'))

# Compile the model you just created using
# -> adam optimizer as optimizer
# -> sparse categorical crossentropy as loss function
# -> accuracy as metric
model.compile(optimizer=adam, loss=loss[0], metrics=[metrics[0]])

# Fit your model and save the returned value as "history". Set properly both the batch size value and the epochs value.
# Use both the train and validation set.
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs)

# Call the plot_history function to plot the obtained results
plot_history(history)

# Evaluate
_, train_acc = model.evaluate(x_train, y_train, verbose=1)
_, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))

# ### 1.3.2 Dropout

# Dropout is one of the most effective and most commonly used regularization techniques for neural networks.
# Dropout, applied to a layer, consists of randomly "dropping out" (i.e. set to zero) a number of output features of the
# layer during training.
# The "dropout rate" is the fraction of the features that are being zeroed-out; it is usually set between 0.2 and 0.5;
# at test time no units are dropped out, and instead the layer's output values are scaled down by a factor equal to the
# dropout rate, so as to balance for the fact that more units are active than at training time.

# Build the model
model = Sequential()
model.add(Flatten())
# Add a Dense layer with 512 neurons, with relu as activation function
model.add(Dense(512, activation='relu'))
# Add a Dropout layer with 0.3 drop probability
model.add(Dropout(0.3))
# Add a Dense layer with 256 neurons, with relu as activation function
model.add(Dense(256, activation='relu'))
# Add a Dropout layer with 0.3 drop probability
model.add(Dropout(0.3))
# Add a Dense layer with 128 neurons, with relu as activation function
model.add(Dense(128, activation='relu'))
# Add a Dropout layer with 0.3 drop probability
model.add(Dropout(0.3))
# Add a Dense layer with 64 neurons, with relu as activation function
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# Add a Dense layer with number of neurons equal to the number of classes, with softmax as activation function
model.add(Dense(num_classes, activation='softmax'))

# Compile the model you just created using
# -> adam optimizer as optimizer
# -> sparse categorical crossentropy as loss function
# -> accuracy as metric
model.compile(optimizer=adam, loss=loss[0], metrics=[metrics[0]])

# Fit your model and save the returned value as "history". Set properly both the batch size value and the epochs value.
# Use both the train and validation set.
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs)

# Call the plot_history function to plot the obtained results
plot_history(history)

# Evaluate
_, train_acc = model.evaluate(x_train, y_train, verbose=1)
_, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))

# ### 1.3.3 Early stopping

# Validation can be used to detect when overfitting starts during supervised training of a neural network; training is
# then stopped before convergence to avoid the overfitting.
os.mkdir('my_checkpoint_dir')
# Early stopping:  https://keras.io/callbacks/
# Note: Super patient dropout scheme is of course equivalent to no-drop out. A super impatient dropout will be
# equivalent to stopping in the first few epochs: this is problematic since it depends on initializations + underfitting
es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# Create checkpoint callback that will save the best model observed during training for later use
checkpoint_path = "my_checkpoint_dir/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)

# Build the model
model = Sequential()
model.add(Flatten())
# Add a Dense layer with 512 neurons, with relu as activation function
model.add(Dense(512, activation='relu'))
# Add a Dense layer with 256 neurons, with relu as activation function
model.add(Dense(256, activation='relu'))
# Add a Dense layer with 128 neurons, with relu as activation function
model.add(Dense(128, activation='relu'))
# Add a Dense layer with 64 neurons, with relu as activation function
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# Add a Dense layer with number of neurons equal to the number of classes, with softmax as activation function
model.add(Dense(num_classes, activation='softmax'))

# Compile the model you just created using
# -> adam optimizer as optimizer
# -> sparse categorical crossentropy as loss function
# -> accuracy as metric
model.compile(optimizer=adam, loss=loss[0], metrics=[metrics[0]])

# Fit your model and save the returned value as "history". Set properly both the batch size value and the epochs value.
# Use both the train and validation set. Be careful to also set properly the callbacks parameter list.
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs,
                    verbose=1, callbacks=[es_callback, cp_callback])

# Call the plot_history function to plot the obtained results
plot_history(history)

# Evaluate
_, train_acc = model.evaluate(x_train, y_train, verbose=1)
_, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))

# ## Load weights:
# The saved weights can then be loaded and evaluated any time by calling the load_weights() function.
# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook) are in place to discourage outdated usage, and can be
# ignored. Link https://www.tensorflow.org/tutorials/keras/save_and_restore_models
model.load_weights(checkpoint_path)
