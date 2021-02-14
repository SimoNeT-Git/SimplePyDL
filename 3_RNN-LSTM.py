# Lab 3: Deep Sequence Modeling

# Main parts:
#    3.1 Deal with sequential data
#    3.2 Recurrent Neural Network
#    3.3 LSTM Network
#    3.4 Reuters newswire classification dataset

# The third lab session is about data that have a sequential structure that must be taken into account.

from __future__ import print_function
import tensorflow as tf
import os, json, re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# ----------------------- 3.1 Deal with sequential data -----------------------

# In this lab we see Deep Learning models that can process sequential data (text, timeseries,..).
# These models don’t take as input raw text: they only work with numeric tensors; vectorizing text is the process of
# transforming text into numeric tensors.
# The different units into which you can break down text (words, characters) are called tokens; then if you apply a
# tokenization scheme, you associate numeric vectors with the generated tokens. These vectors, packed into sequence
# tensors, are fed into Deep Neural Network. There are multiple ways to associate a vector with a token: we will see
# One-Hot Encoding and Token Embedding.
# In this section we are going to deal with:
#     3.1.1 One-Hot Encoding
#     3.1.2 Word embedding

# ### 3.1.1 One-Hot Encoding

# One-Hot Encoding consists of associating a unique integer index with every word and then turning this integer
# index i into a binary vector of size N (the size of the vocabulary); the vector is all zeros except for the
# i-th entry, which is 1.

# Try to perform One-Hot Encoding using Tokenizer.
# Keras provides the Tokenizer class for preparing text documents for DL.
# The Tokenizer must be constructed and then fit on either raw text documents or integer encoded text documents.
# define 4 documents
docs = ['Well done!', 'Good work', 'Great effort', 'Nice work']

# create the tokenizer
tokenizer = Tokenizer()

# fit the tokenizer on the documents
tokenizer.fit_on_texts(docs)
encoded_docs = tokenizer.texts_to_matrix(docs, mode='count')
print("Tokenization with one-hot encoding on the 'docs' list gives:")
print(encoded_docs)

# Some problems related to this kind of encoding are sparsity of the solution and the high dimensionality of the vector
# encoding of the tokens.

# ### 3.1.2 Word embedding

# The vector obtained from word embedding is dense and has lower dimensionality w.r.t One-Hot Encoding vector;
# the dimensionality of embedding space vector is an hyperparameter.
# There are two ways to obtain word embeddings:
#    - May be learned jointly with the network
#    - May use pre-trained word vectors (Word2Vec, GloVe,..)
# Word embeddings maps human language into a geometric space; in a reasonable embedding space synonyms are embedded
# into similar word vectors and the geometric distance between any two word vectors reflects the semantic distance
# between the associated words (words meaning different things are embedded at points far away from each other,
# whereas related words are closer).
# How good is a word-embedding space depends on the specific task. It is reasonable to learn a new embedding space with
# every new task: with backpropagation and Keras it reduces to learn the weights of the Embedding layer.

# Learning Word Embeddings with the embedding layer.
# Load imdb dataset: This dataset contains movies reviews from IMDB, labeled by sentiment(positive/negative); reviews
# have been preprocessed, and each review is encoded as a sequence of word indexes (integers).
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
max_features = 10000
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Show the size of vocabulary and the most frequent words
word_to_index = imdb.get_word_index()
vocab_size = len(word_to_index)
print('Vocab size : ', vocab_size)
words_freq_list = []
for (k, v) in word_to_index.items():
    words_freq_list.append((k, v))
sorted_list = sorted(words_freq_list, key=lambda x: x[1])
print("50 most common words:")
print(sorted_list[0:50])

# Converting IMDB dataset to readable reviews.
# Reviews in the IMDB dataset have been encoded as a sequence of integers. The dataset also contains an index for
# converting the reviews back into human readable form.

# Get the word index from the dataset
word_index = {k: v for k, v in word_to_index.items()}

# Perform reverse word lookup and make it callable
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# Data Insight
# Map for readable classnames
class_names = ["Negative", "Positive"]

# Concatenate test and training datasets
allreviews = np.concatenate((x_train, x_test), axis=0)

# Review lengths across test and training whole datasets
print("Maximum review length: {}".format(len(max(allreviews, key=len))))
print("Minimum review length: {}".format(len(min(allreviews, key=len))))
result = [len(x) for x in allreviews]
print("Mean review length: {}".format(np.mean(result)))

# Print a review and it's class as stored in the dataset. Replace the number
# to select a different review.
print("\nMachine readable Review")
print("  Review Text: " + str(x_train[3]))
print("  Review Sentiment: " + str(y_train[3]))

# Print a review and it's class in human readable format. Replace the number
# to select a different review.
print("\nHuman Readable Review")
print("  Review Text: " + decode_review(x_train[3]))
print("  Review Sentiment: " + class_names[y_train[3]])

# Pre-processing Data.
# We need to make sure that our reviews are of a uniform length. Some reviews will need to be truncated, while others
# need to be padded.
maxlen = 50
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print("Shape Training  Data: " + str(x_train.shape))
print("Shape Training Output Data: " + str(y_train.shape))
print("Shape Test Data: " + str(x_test.shape))
print("Shape Test Output Data: " + str(y_test.shape))


# ----------------------- 3.2 Recurrent Neural Network -----------------------

# Here https://colah.github.io/posts/2015-08-Understanding-LSTMs/ you can find a clear explanation about RNNs and LSTMs;
# the following is a summary of the main concepts.
# A major characteristic of some neural networks, as ConvNet, is that they have no memory: each input is processed
# independently, with no state kept in between inputs. Biological intelligence processes information incrementally while
# maintaining an internal model of what it’s processing, built from past information and constantly updated as new
# information comes in.
# A recurrent neural network (RNN) adopts the same principle but in an extremely simplified version: it processes
# sequences by iterating through the sequence elements and maintaining a state containing information relative to what
# it has seen so far.
# Each input x[i=t−1,t,t+1,..] is combined with the internal state and then is applied an activation function
# (e.g. tanh) then the output is computed h[i=t−1,t,t+1,..] and the internal state is updated.
# In many cases, you just need the last output (h[i=last] at the end of the loop), because it already contains
# information about the entire sequence.

# Create the model.
# In the following sections we will develop different models. Be careful to the fact that we are dealing with a binary
# classification problem!
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 16))
model.add(tf.keras.layers.Dropout(rate=0.2))  # Randomly disable 20% of neurons
# Complete the model, it should be made by at least:
# 1 x SimpleRNN layer (20% dropout for input units and 20% dropout for recurrent connections)
# 1 x Dense layer
model.add(tf.keras.layers.SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Train your model here
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


def plot_history(hist):
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


plot_history(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy: %.3f, Test loss: %.3f' % (test_acc, test_loss))

# Classification Report
predicted_classes = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test, predicted_classes, target_names=class_names))

# Try to build a new model where you stack several recurrent layers.
# In such a setup, you have to get all of the intermediate layers to return full sequence of outputs. This is needed to
# return batch size, timesteps, hidden state. By doing this the output should contain all historical generated outputs
# along with time stamps (3D). This way the next layer can work further on the data.
# Build the model. It should be made by at least:
# 1 x Embedding layer
# > 1 x SimpleRNN layer (do not forget to put the return_sequences parameter to True)
# 1 x Dense layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 16))
model.add(tf.keras.layers.Dropout(rate=0.2))  # Randomly disable 20% of neurons
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Train your model here
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
plot_history(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy: %.3f, Test loss: %.3f' % (test_acc, test_loss))

predicted_classes = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test, predicted_classes, target_names=class_names))

# What can you say about the obtained results? What about the comparison between these results and the ones obtained in
# the single layer RNN?


# ----------------------- 3.3 LSTM Network -----------------------

# LSTMs are a special kind of recurrent neural network which works, for many tasks, much better than the standard RNNs.
# These nets are capable of learning long-term dependencies (they are explicitly designed to avoid the long-term
# dependency problem); remembering information for long periods of time is practically their default behavior.
# RNNs have a very simple structure, such as a single tanh layer.
# LSTMs also have a chain like structure, but the repeating module has a different structure: instead of having a single
# neural network layer, there are four, interacting in a very special way.

# Create LSTM model in TensorFlow

# Build the model. It should be made by at least:
# 1 x Embedding layer
# 1 x LSTM layer
# 1 x Dense layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=max_features, output_dim=32))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.LSTM(32, dropout=0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Train your model here
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

plot_history(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy: %.3f, Test loss: %.3f' % (test_acc, test_loss))

predicted_classes = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test, predicted_classes, target_names=class_names))

# ----------------------- 3.4 Reuters newswire classification dataset -----------------------

# The reuters newswire classification dataset is a dataset of 11,228 newswires from Reuters, labeled over 46 topics.
# More information about the dataset and how to use it can be found here: https://keras.io/api/datasets/reuters/
# Try to build a new model dealing with this new dataset. Try to use both the RNN and the LSTM approach, and select the
# best of them. What do you expect will be the best? Be carefull that this domain shift will imply some changes in your
# code as it is not a binary classification problem anymore!

# Load Reuters dataset
max_features = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=max_features)

# Data Insight
class_names = ['cocoa', 'grain', 'veg-oil', 'earn', 'acq', 'wheat', 'copper', 'housing', 'money-supply',
               'coffee', 'sugar', 'trade', 'reserves', 'ship', 'cotton', 'carcass', 'crude', 'nat-gas',
               'cpi', 'money-fx', 'interest', 'gnp', 'meal-feed', 'alum', 'oilseed', 'gold', 'tin',
               'strategic-metal', 'livestock', 'retail', 'ipi', 'iron-steel', 'rubber', 'heat', 'jobs',
               'lei', 'bop', 'zinc', 'orange', 'pet-chem', 'dlr', 'gas', 'silver', 'wpi', 'hog', 'lead']

num_classes = np.max(y_train) + 1
print('Number of Training Samples: {}'.format(len(x_train)))
print('Number of Test Samples: {}'.format(len(x_test)))
print('Number of classes: ', np.max(y_train) + 1)  # plus one because indexing of categories starts at 0

# Concatonate test and training datasets
allreviews = np.concatenate((x_train, x_test), axis=0)

# Review lengths across test and training whole datasets
print("Maximum review length: {}".format(len(max((allreviews), key=len))))
print("Minimum review length: {}".format(len(min((allreviews), key=len))))
result = [len(x) for x in allreviews]
print("Mean review length: {}".format(np.mean(result)))

# Print a review and it's class as stored in the dataset. Replace the number
# to select a different review.
print("")
print("Machine readable Review")
print("  Review Text: " + str(x_train[3]))
print("  Review Sentiment: " + str(y_train[3]))

# Print a review and it's class in human readable format. Replace the number
# to select a different review.
print("")
print("Human Readable Review")
print("  Review Text: " + decode_review(x_train[3]))
print("  Review Sentiment: " + class_names[y_train[3]])

maxlen = 50
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# ### RNN

# Build the model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 16))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.SimpleRNN(32, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
y_train_cat = tf.keras.utils.to_categorical(y_train)

# Train your model here
history = model.fit(x_train, y_train_cat,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.15)

plot_history(history)

# Evaluate the model
y_test_cat = tf.keras.utils.to_categorical(y_test)
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=1)
print('Test accuracy: %.3f, Test loss: %.3f' % (test_acc, test_loss))

rounded_y_test = np.argmax(y_test_cat, axis=1)
predicted_classes = model.predict_classes(x_test)
print(confusion_matrix(rounded_y_test, predicted_classes))
print(classification_report(rounded_y_test, predicted_classes, target_names=class_names))

# ### LSTM

# Build the model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 16))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2))
model.add(tf.keras.layers.LSTM(32, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
y_train = tf.keras.utils.to_categorical(y_train)

# Train your model here
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.15)

plot_history(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=1)
print('Test accuracy: %.3f, Test loss: %.3f' % (test_acc, test_loss))

rounded_y_test = np.argmax(y_test_cat, axis=1)
predicted_classes = model.predict_classes(x_test)
print(confusion_matrix(rounded_y_test, predicted_classes))
print(classification_report(rounded_y_test, predicted_classes, target_names=class_names))

print("Not predicted classes: ", set(y_test) - set(predicted_classes))
# This means that there is no F-score to calculate for this label, thus the F-score for this case is considered to be 0

# Are the results accordant to what you expected? Can you notice some differences between the RNN and the LSTM results?
# Why?
