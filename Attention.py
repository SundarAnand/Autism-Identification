import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img
from keras import backend as K
from sklearn.metrics import classification_report
import seaborn as sns

# essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
from matplotlib import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support

# tensorflow, keras
import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation, Add, Multiply
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img

# essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
# from matplotlib.pyplot import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# attention needed
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Reshape
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random
import sys
from numpy import load
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def AttentionConvNet(input_shape):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # First Block
    # CONV layer
    conv_1_out = Conv2D(32, (7, 7), strides=(1, 1), activation='relu', padding='same')(X_input)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2, 2), strides=2, padding='same')(conv_1_out)
    X = BatchNormalization(axis=-1)(X)

    block_2_in1_conv = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_1_out)
    block_2_in1_max = MaxPooling2D((2, 2), strides=1, padding='same')(block_2_in1_conv)

    # attention
    dense_from_block_1 = densor_block2(block_2_in1_max)
    activator_from_block_1 = activator(dense_from_block_1)
    dotProduct = mult([activator_from_block_1, block_2_in1_max])

    attention_input_2 = Add()([dotProduct, X])

    # Second Block
    # CONV layer
    conv_2_out = Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same')(attention_input_2)
    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=2, padding='same')(conv_2_out)
    X = BatchNormalization(axis=-1)(X)

    block_3_in2_conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv_2_out)
    block_3_in2_max = MaxPooling2D((2, 2), strides=2, padding='same')(block_3_in2_conv)

    block_3_in1_conv = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(block_2_in1_max)
    block_3_in1_max = MaxPooling2D((2, 2), strides=2, padding='same')(block_3_in1_conv)

    # attention
    dense_from_block_1 = densor_block3(block_3_in1_max)
    activator_from_block_1 = activator(dense_from_block_1)
    dotProduct_1 = mult([activator_from_block_1, block_3_in1_max])

    # attention
    dense_from_block_2 = densor_block3(block_3_in2_max)
    activator_from_block_2 = activator(dense_from_block_2)
    dotProduct_2 = mult([activator_from_block_2, block_3_in2_max])

    attention_input_3 = Add()([dotProduct_1, dotProduct_2, X])

    # Third Block
    conv_3_out = Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding='same')(attention_input_3)
    # MAXPOOL
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv_3_out)
    X = BatchNormalization(axis=-1)(X)
    # Top layer
    X = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Conv2D(64, (7, 7), strides=(2, 2), activation='relu')(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=-1))(X)

    # Final output layer. First Unit is a sigmoid act(whether seen img is infected/not)
    # next 2 units for identifying type of infection if 1st element is 1. otherwise, don't care.
    Emotion = Conv2D(2, (1, 1), strides=(1, 1), activation='softmax')(X)
    reshapeOut = Reshape((2,))(Emotion)

    model = Model(inputs=X_input, outputs=reshapeOut)

    return model

# load train and test dataset
def load_dataset():
	# load dataset
	data = load('ASD.npz')
	x, y = data['arr_0'], data['arr_1']
	# separate into train and test datasets
	X, testX, Y, testY = train_test_split(x, y, test_size=0.3, random_state=1)

	print(X.shape, Y.shape, testX.shape, testY.shape)
	return x, y, X, Y, testX, testY

def val_split(X, Y):
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=1)
    print(trainX.shape, trainY.shape, valX.shape, valY.shape)
    return trainX, trainY, valX, valY 

#function to draw confusion matrix
def draw_confusion_matrix(true,preds):
    conf_matx = confusion_matrix(true, preds)
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="OrRd")
    plt.savefig('Heatmap.png')
    #return conf_matx


##Using the GPU

with tf.device('/device:GPU:0'):
    tf.reset_default_graph()

    # Defined shared layers as global variables
    concatenator = Concatenate(axis=-1)
    densor_block2 = Dense(1, activation="relu")
    densor_block3 = Dense(1, activation="relu")
    activator = Activation('softmax',
                           name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=2)
    mult = Multiply()

    ## Initializing the model
    new_model = AttentionConvNet((256, 256, 3))
    #history=new_model.load_weights("final_model.h5")
    ## Compling the model
    new_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    new_model.summary()
    # fit model

    #compile model using accuracy to measure model performance
    #new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    FullX, FullY, trainX, trainY, testX, testY = load_dataset()

    trainX, trainY, valX, valY = val_split(X, Y)
    #train the model
    history = new_model.fit(trainX, trainY, validation_data=(valX, valY), epochs=20)
    new_model.save('Final_model_VIT.h5')

    print (testY)
    testY_bool = np.argmax(testY, axis=1)
    print (testY_bool)

    y_pred = new_model.predict(testX, batch_size=64, verbose=1)
    print (y_pred)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print (y_pred_bool)


    results = confusion_matrix(testY_bool, y_pred_bool) 
    print ('Confusion Matrix :')
    print(results) 
    print ('Report : ')
    print(classification_report(testY_bool, y_pred_bool))

    print("Accuracy : ")
    print (accuracy_score(testY_bool, y_pred_bool))

    print("Precision : ")
    print(precision_score(testY_bool, y_pred_bool, average='macro'))

    print("Recall : ")
    print(recall_score(testY_bool, y_pred_bool, average='macro'))

    print("F1_Score : ")
    print(f1_score(testY_bool, y_pred_bool, average='macro'))

    draw_confusion_matrix(testY_bool, y_pred_bool)

    print (precision_recall_fscore_support(testY_bool, y_pred_bool, average='macro'))
