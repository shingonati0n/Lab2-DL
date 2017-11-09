#Definicion de librerias con la funciones que seran utilizadas por Keras.
import keras
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

#Definicion de contenedor y primera capa de AlexNet.
layerCount = 1
modelAlexNet = Sequential()
modelAlexNet.add(ZeroPadding2D((2,2), input_shape=(224, 224, 3)))
modelAlexNet.add(Conv2D(96,(11,11),strides=(4,4),padding='valid'))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(BatchNormalization())
print("Batch Normalization Capa {}: {} ".format(layerCount,modelAlexNet.output_shape))
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print("Max Pooling Capa {}: {}".format(layerCount,modelAlexNet.output_shape))
layerCount = layerCount + 1
#Capa 2
modelAlexNet.add(ZeroPadding2D((2,2)))
modelAlexNet.add(Conv2D(256, (5, 5), padding='valid'))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(BatchNormalization())
print("Batch Normalization Capa {}: {} ".format(layerCount,modelAlexNet.output_shape))
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print("Max Pooling Capa {}: {}".format(layerCount,modelAlexNet.output_shape))
layerCount = layerCount + 1
#Capa 3
modelAlexNet.add(ZeroPadding2D((1,1)))
modelAlexNet.add(Conv2D(384, (3, 3), padding='valid'))
modelAlexNet.add(Activation(activation='relu'))
print("Convolución Capa {}: {} ".format(layerCount,modelAlexNet.output_shape))
layerCount = layerCount + 1
#Capa 4
modelAlexNet.add(ZeroPadding2D((1,1)))
modelAlexNet.add(Conv2D(384, (3, 3), padding='valid'))
modelAlexNet.add(Activation(activation='relu'))
print("Convolución Capa {}: {} ".format(layerCount,modelAlexNet.output_shape))
layerCount = layerCount + 1
#Capa 5
modelAlexNet.add(ZeroPadding2D((1,1)))
modelAlexNet.add(Conv2D(256,(3,3),padding='valid'))
modelAlexNet.add(Activation(activation='relu'))
print("Convolución Capa {}: {} ".format(layerCount,modelAlexNet.output_shape))
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print("Max Pooling Capa {}: {}".format(layerCount,modelAlexNet.output_shape))
#Representar output de capa 5 a 1D usando flatten
modelAlexNet.add(Flatten())
print("Representación 1D Capa {}: {}".format(layerCount,modelAlexNet.output_shape))
#Capa 6
modelAlexNet.add(Dense(units=4096, input_dim=(9216),kernel_initializer='glorot_uniform'))
modelAlexNet.add(Activation('relu'))
modelAlexNet.add(Dropout(0.5))
layerCount = layerCount + 1
print("Capa Densa {}: {}".format(layerCount,modelAlexNet.output_shape))
#Capa 7
modelAlexNet.add(Dense(units=4096))
modelAlexNet.add(Activation('relu'))
modelAlexNet.add(Dropout(0.5))
layerCount = layerCount + 1
print("Capa Densa {}: {}".format(layerCount,modelAlexNet.output_shape))
#Capa 8
modelAlexNet.add(Dense(units=1000))
modelAlexNet.add(Activation('softmax'))
layerCount = layerCount + 1
print("Capa Densa {}: {}".format(layerCount,modelAlexNet.output_shape))
modelAlexNet.summary()

