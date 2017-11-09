#Actividad 2

#Siguiendo con la arquitectura de AlexNet en la figura 1 y las funciones definidas anteriormente, la segunda capa queda definida por:

#modelAlexNet.add(ZeroPadding2D((2,2)))
#modelAlexNet.add(Convolution2D(256, 5, 5, border_mode=’valid’))
#modelAlexNet.add(Activation(activation=’relu ’))
#modelAlexNet.add(BatchNormalization())
#modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))

#Verifique que las dimensiones de salida de esta segunda capa corresponden a las utilizadas por AlexNet.
#En su informe de laboratorio incorporé los output generados. Adicionalmente, utilice el comando modelAlexNet.summary() para generar un resumen de la red construida hasta este momento. ¿Cuántos parámetros (pesos) contiene esta red?

#Codigo ejecutado
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
modelAlexNet.summary()

#Output
#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayer2.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
#Batch Normalization Capa 1: (None, 55, 55, 96) 
#Max Pooling Capa 1: (None, 27, 27, 96)
#Batch Normalization Capa 2: (None, 27, 27, 256) 
#Max Pooling Capa 2: (None, 13, 13, 256)
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#zero_padding2d_2 (ZeroPaddin (None, 228, 228, 3)       0         
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 55, 55, 96)        34944     
#_________________________________________________________________
#activation_2 (Activation)    (None, 55, 55, 96)        0         
#_________________________________________________________________
#batch_normalization_2 (Batch (None, 55, 55, 96)        384       
#_________________________________________________________________
#max_pooling2d_2 (MaxPooling2 (None, 27, 27, 96)        0         
#_________________________________________________________________
#zero_padding2d_3 (ZeroPaddin (None, 31, 31, 96)        0         
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, 27, 27, 256)       614656    
#_________________________________________________________________
#activation_3 (Activation)    (None, 27, 27, 256)       0         
#_________________________________________________________________
#batch_normalization_3 (Batch (None, 27, 27, 256)       1024      
#_________________________________________________________________
#max_pooling2d_3 (MaxPooling2 (None, 13, 13, 256)       0         
#=================================================================
#Total params: 651,008
#Trainable params: 650,304
#Non-trainable params: 704
#
#Las dimensiones corresponden a AlexNet. La cantidad total de parametros hasta el momento es de 651008, contando los parametros generados por BatchNormalization.
#Cabe destacar que la cantidad real de pesos serian aquellos de la primera capa: 34944 (34848 + 1 bias vector de 96) mas los de la segunda capa: 614656 (614400 + 1 bias vector de 256)  