# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:52:29 2019

@author: Raval
"""

import tflearn
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from tflearn.data_utils import to_categorical, pad_sequences

data = keras.datasets.imdb


# IMDB load dataset
train, test= data.load_data(num_words=10000) # valid_portion=0.1 (10%) it hepls preventing overfitting
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX =pad_sequences(trainX, maxlen = 100) # pad_sequences is it converts the each review in matrix and padding it add 0 from all it's border.padding necessary for input consistency and dimentionality.
testX = pad_sequences(trainX, maxlen = 100)
# max_length =100 iswords of length or we can change to 256 or 200 to increase accuracy.

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2) # nb_classes=2 whether it is positive or negative
testY = to_categorical(testY, nb_classes=2)

# create neural network
model = Sequential()
model.add(Embedding(10000, 16)) # it embed a value to ex. 0 value when embedding it's like 0:[0.2,0.3,0.4] and it goes to 16 dimension
                                                    # ex. 7 value when embedding it's like 7:[7,7.3,9] and it  goes to 16 dimension              
model.add(GlobalAveragePooling1D())# it take the  average of the embedding layer so it can reduce dimensionality
model.add(Dense(16, activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.summary()


model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit(trainX, trainY,batch_size=32, epochs = 10, validation_data=(testX,testY))


# for new test data accuracy
results = model.evaluate(testX, testY)

print(results)