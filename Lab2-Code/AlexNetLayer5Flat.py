#Actividad 5
#Ha sido una tarea ardua pero estamos cerca de completar la implementación de AlexNet, sólo nos queda la
#definición de las capas de conexión densa (multilayer perceptron o MLP).
#Para la definición de la primera de estas capas, el primer paso es llevar a una representación 1D la salida
#3D de la capa 5 (representada en figura 1 con un cubo). Para esto utilizamos la función de Keras Flatten
#(aplanar), según la siguiente sintaxis:
#modelAlexNet.add(Flatten())
#Utilice la función output shape para verificar el efecto de la función Flatten. ¿Las dimensiones obtenidas
#corresponden a lo esperado?. Fundamente sus observaciones.

#Codigo ejecutado:

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
modelAlexNet.summary()

#Output: 
#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayer5Flat.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
#Batch Normalization Capa 1: (None, 55, 55, 96) 
#Max Pooling Capa 1: (None, 27, 27, 96)
#Batch Normalization Capa 2: (None, 27, 27, 256) 
#Max Pooling Capa 2: (None, 13, 13, 256)
#Convolución Capa 3: (None, 13, 13, 384) 
#Convolución Capa 4: (None, 13, 13, 384) 
#Convolución Capa 5: (None, 13, 13, 256) 
#Max Pooling Capa 5: (None, 6, 6, 256)
#Representación 1D Capa 5: (None, 9216)
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#zero_padding2d_15 (ZeroPaddi (None, 228, 228, 3)       0         
#_________________________________________________________________
#conv2d_15 (Conv2D)           (None, 55, 55, 96)        34944     
#_________________________________________________________________
#activation_15 (Activation)   (None, 55, 55, 96)        0         
#_________________________________________________________________
#batch_normalization_10 (Batc (None, 55, 55, 96)        384       
#_________________________________________________________________
#max_pooling2d_11 (MaxPooling (None, 27, 27, 96)        0         
#_________________________________________________________________
#zero_padding2d_16 (ZeroPaddi (None, 31, 31, 96)        0         
#_________________________________________________________________
#conv2d_16 (Conv2D)           (None, 27, 27, 256)       614656    
#_________________________________________________________________
#activation_16 (Activation)   (None, 27, 27, 256)       0         
#_________________________________________________________________
#batch_normalization_11 (Batc (None, 27, 27, 256)       1024      
#_________________________________________________________________
#max_pooling2d_12 (MaxPooling (None, 13, 13, 256)       0         
#_________________________________________________________________
#zero_padding2d_17 (ZeroPaddi (None, 15, 15, 256)       0         
#_________________________________________________________________
#conv2d_17 (Conv2D)           (None, 13, 13, 384)       885120    
#_________________________________________________________________
#activation_17 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_18 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_18 (Conv2D)           (None, 13, 13, 384)       1327488   
#_________________________________________________________________
#activation_18 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_19 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_19 (Conv2D)           (None, 13, 13, 256)       884992    
#_________________________________________________________________
#activation_19 (Activation)   (None, 13, 13, 256)       0         
#_________________________________________________________________
#max_pooling2d_13 (MaxPooling (None, 6, 6, 256)         0         
#_________________________________________________________________
#flatten_1 (Flatten)          (None, 9216)              0         
#=================================================================
#Total params: 3,748,608
#Trainable params: 3,747,904
#Non-trainable params: 704
#
#Las dimensiones corresponden a lo esperado. En caso de Flatten() es facilmente visible; las dimensiones de salida de la capa 5 son (6,6,256). 
#En este punto Flatten() funciona como un FlatMap. 9216 = 6*6*256