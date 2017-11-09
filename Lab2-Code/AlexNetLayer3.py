#Actividad 3

#Usando como guı́a las capas anteriores, construya en Keras la tercera capa de AlexNet. Tenga presente que esta tercera capa:
#• Incluye un padding de 1 cero a cada lado del mapa de activaciones de entrada.
#• No utiliza normalización batch.
#• No incorpora una etapa de max-pooling.
#En su informe de laboratorio reporte el código generado. Adicionalmente, utilice la función output shape para verificar que la salida de la tercera capa 
#corresponde a la arquitectura de la figura 1, y la función summary para obtener un resumen de los parámetros de la red.

#Codigo ejecutado para capa 3:

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
modelAlexNet.summary()

#Output:
#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayer3.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
#Batch Normalization Capa 1: (None, 55, 55, 96) 
#Max Pooling Capa 1: (None, 27, 27, 96)
#Batch Normalization Capa 2: (None, 27, 27, 256) 
#Max Pooling Capa 2: (None, 13, 13, 256)
#Convolución Capa 3: (None, 13, 13, 384) 
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#zero_padding2d_7 (ZeroPaddin (None, 228, 228, 3)       0         
#_________________________________________________________________
#conv2d_7 (Conv2D)            (None, 55, 55, 96)        34944     
#_________________________________________________________________
#activation_7 (Activation)    (None, 55, 55, 96)        0         
#_________________________________________________________________
#batch_normalization_6 (Batch (None, 55, 55, 96)        384       
#_________________________________________________________________
#max_pooling2d_6 (MaxPooling2 (None, 27, 27, 96)        0         
#_________________________________________________________________
#zero_padding2d_8 (ZeroPaddin (None, 31, 31, 96)        0         
#_________________________________________________________________
#conv2d_8 (Conv2D)            (None, 27, 27, 256)       614656    
#_________________________________________________________________
#activation_8 (Activation)    (None, 27, 27, 256)       0         
#_________________________________________________________________
#batch_normalization_7 (Batch (None, 27, 27, 256)       1024      
#_________________________________________________________________
#max_pooling2d_7 (MaxPooling2 (None, 13, 13, 256)       0         
#_________________________________________________________________
#zero_padding2d_9 (ZeroPaddin (None, 15, 15, 256)       0         
#_________________________________________________________________
#conv2d_9 (Conv2D)            (None, 13, 13, 384)       885120    
#_________________________________________________________________
#activation_9 (Activation)    (None, 13, 13, 384)       0         
#=================================================================
#Total params: 1,536,128
#Trainable params: 1,535,424
#Non-trainable params: 704