#Actividad 6
#Como vimos en clases, las capas densas son definidas en Keras mediante la función Dense(). Revise sus
#apuntes y genere el código apropiado para definir la primera capa densa que tiene como salida 4096 neu-
#ronas. Tal como en las capas convolucionales, AlexNet utiliza para la capa 6 una función de activación Relu.
#Adicionalmente, incorpora un proceso de Dropout con una probabilidad de 0.5.
#3Revise sus apuntes de clases, donde se ilustra el uso de Dropout en Keras, y genere el código restante para
#completar la definición de la capa 6 de AlexNet. Verifique que las salidas sean las adecuadas, y determine el
#número de parámetros de la red. Reporte sus observaciones y código generado.

#Codigo Ejecutado
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
modelAlexNet.summary()

#Output:
#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayer6.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
#Batch Normalization Capa 1: (None, 55, 55, 96) 
#Max Pooling Capa 1: (None, 27, 27, 96)
#Batch Normalization Capa 2: (None, 27, 27, 256) 
#Max Pooling Capa 2: (None, 13, 13, 256)
#Convolución Capa 3: (None, 13, 13, 384) 
#Convolución Capa 4: (None, 13, 13, 384) 
#Convolución Capa 5: (None, 13, 13, 256) 
#Max Pooling Capa 5: (None, 6, 6, 256)
#Representación 1D Capa 5: (None, 9216)
#Capa Densa 6: (None, 4096)
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#zero_padding2d_20 (ZeroPaddi (None, 228, 228, 3)       0         
#_________________________________________________________________
#conv2d_20 (Conv2D)           (None, 55, 55, 96)        34944     
#_________________________________________________________________
#activation_20 (Activation)   (None, 55, 55, 96)        0         
#_________________________________________________________________
#batch_normalization_12 (Batc (None, 55, 55, 96)        384       
#_________________________________________________________________
#max_pooling2d_14 (MaxPooling (None, 27, 27, 96)        0         
#_________________________________________________________________
#zero_padding2d_21 (ZeroPaddi (None, 31, 31, 96)        0         
#_________________________________________________________________
#conv2d_21 (Conv2D)           (None, 27, 27, 256)       614656    
#_________________________________________________________________
#activation_21 (Activation)   (None, 27, 27, 256)       0         
#_________________________________________________________________
#batch_normalization_13 (Batc (None, 27, 27, 256)       1024      
#_________________________________________________________________
#max_pooling2d_15 (MaxPooling (None, 13, 13, 256)       0         
#_________________________________________________________________
#zero_padding2d_22 (ZeroPaddi (None, 15, 15, 256)       0         
#_________________________________________________________________
#conv2d_22 (Conv2D)           (None, 13, 13, 384)       885120    
#_________________________________________________________________
#activation_22 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_23 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_23 (Conv2D)           (None, 13, 13, 384)       1327488   
#_________________________________________________________________
#activation_23 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_24 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_24 (Conv2D)           (None, 13, 13, 256)       884992    
#_________________________________________________________________
#activation_24 (Activation)   (None, 13, 13, 256)       0         
#_________________________________________________________________
#max_pooling2d_16 (MaxPooling (None, 6, 6, 256)         0         
#_________________________________________________________________
#flatten_2 (Flatten)          (None, 9216)              0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 4096)              37752832  
#_________________________________________________________________
#activation_25 (Activation)   (None, 4096)              0         
#_________________________________________________________________
#dropout_1 (Dropout)          (None, 4096)              0         
#=================================================================
#Total params: 41,501,440
#Trainable params: 41,500,736
#Non-trainable params: 704

#La funcion Dense tambien reporta sus parametros sumando un bias vector de 4096. Esto ocurre dado a que Keras toma como True el valor para el parametro use_bias cuando este no es
#declarado. Cabe destacar tambien que como es la primera instancia de Dense, se debe declarar la dimension de entrada, en este caso se utilizo la de la capa plana anterior (9216). 
#Tambien era posible utilizar las dimensiones separadas en el parametro input_dim, en este caso (6,6,256). Keras toma los parametros y los multiplica contra el tamaño 
#de la capa densa.