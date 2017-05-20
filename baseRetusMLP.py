#!/bin/bash/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'wolker'

'''
Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
8083/8083 [==============================] - 3s - loss: 1.4323 - acc: 0.6775 - val_loss: 1.0948 - val_acc: 0.7631
Epoch 2/5
8083/8083 [==============================] - 3s - loss: 0.7900 - acc: 0.8168 - val_loss: 0.9400 - val_acc: 0.7875
Epoch 3/5
8083/8083 [==============================] - 3s - loss: 0.5511 - acc: 0.8648 - val_loss: 0.8944 - val_acc: 0.8009
Epoch 4/5
8083/8083 [==============================] - 3s - loss: 0.4158 - acc: 0.8972 - val_loss: 0.8790 - val_acc: 0.8053
Epoch 5/5
8083/8083 [==============================] - 3s - loss: 0.3251 - acc: 0.9186 - val_loss: 0.9104 - val_acc: 0.7964
1888/2246 [========================>.....] - ETA: 0s
Test score: 0.890569410256
Test accuracy: 0.793410507569
'''


import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
epochs = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')  #8982
print(len(x_test), 'test sequences')    #2246

num_classes = np.max(y_train) + 1       #46
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)  # (8982, 1000)
print('x_test shape:', x_test.shape)    # (2246, 1000)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)  #(8982, 46)
print('y_test shape:', y_test.shape)    #(2246, 46)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])      #0.890569410256
print('Test accuracy:', score[1])   #0.793410507569