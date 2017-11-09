#Actividad 4

#Para completar la fase convolucional de AlexNet nos queda definir las capas 4 y 5. 
#Análogamente a la capa 3, la capa 4 también realiza un padding de 1 cero a cada lado de la entrada, 
#no incluye normalización batch, y no incluye max-pooling. 
#Por su parte, la capa 5 realiza un padding de 1 cero a cada lado de la entrada, 
#no incorpora normalización batch, pero si incorpora una etapa de max-pooling con una ventana de 3x3 
#y un stride de 2 en las direcciones vertical y horizontal.
#Genere el código de las capas 4 y 5, verifique que las salidas sean las adecuadas, 
#y determine el número de parámetros de la red. Reporte sus observaciones y código generado.

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
modelAlexNet.summary()

#Output

#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayers4-5.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
#Batch Normalization Capa 1: (None, 55, 55, 96) 
#Max Pooling Capa 1: (None, 27, 27, 96)
#Batch Normalization Capa 2: (None, 27, 27, 256) 
#Max Pooling Capa 2: (None, 13, 13, 256)
#Convolución Capa 3: (None, 13, 13, 384) 
#Convolución Capa 4: (None, 13, 13, 384) 
#Convolución Capa 5: (None, 13, 13, 256) 
#Max Pooling Capa 5: (None, 6, 6, 256)
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#zero_padding2d_10 (ZeroPaddi (None, 228, 228, 3)       0         
#_________________________________________________________________
#conv2d_10 (Conv2D)           (None, 55, 55, 96)        34944     
#_________________________________________________________________
#activation_10 (Activation)   (None, 55, 55, 96)        0         
#_________________________________________________________________
#batch_normalization_8 (Batch (None, 55, 55, 96)        384       
#_________________________________________________________________
#max_pooling2d_8 (MaxPooling2 (None, 27, 27, 96)        0         
#_________________________________________________________________
#zero_padding2d_11 (ZeroPaddi (None, 31, 31, 96)        0         
#_________________________________________________________________
#conv2d_11 (Conv2D)           (None, 27, 27, 256)       614656    
#_________________________________________________________________
#activation_11 (Activation)   (None, 27, 27, 256)       0         
#_________________________________________________________________
#batch_normalization_9 (Batch (None, 27, 27, 256)       1024      
#_________________________________________________________________
#max_pooling2d_9 (MaxPooling2 (None, 13, 13, 256)       0         
#_________________________________________________________________
#zero_padding2d_12 (ZeroPaddi (None, 15, 15, 256)       0         
#_________________________________________________________________
#conv2d_12 (Conv2D)           (None, 13, 13, 384)       885120    
#_________________________________________________________________
#activation_12 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_13 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_13 (Conv2D)           (None, 13, 13, 384)       1327488   
#_________________________________________________________________
#activation_13 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_14 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_14 (Conv2D)           (None, 13, 13, 256)       884992    
#_________________________________________________________________
#activation_14 (Activation)   (None, 13, 13, 256)       0         
#_________________________________________________________________
#max_pooling2d_10 (MaxPooling (None, 6, 6, 256)         0         
#=================================================================
#Total params: 3,748,608
#Trainable params: 3,747,904
#Non-trainable params: 704
#
#Las dimensiones corresponden a la figura 1 del enunciado. La cantidad de parametros sería 3747904, incluyendo bias vectors para cada layer y batch normalization para 
#las capas 1 y 2 (confirmar!)
