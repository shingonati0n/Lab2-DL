#Definicion de librerias con la funciones que seran utilizadas por Keras.
import keras
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

#Actividad 7
#Defina las capas 7 y 8 de AlexNet. En el caso de la capa 7, como se aprecia en la figura 1, tiene una salida
#de 4096 neuronas. Esta capa utiliza función de activación Relu y Dropout con probabilidad 0.5. Finalmente,
#la capa 8 tiene una salida de 1000 neuronas. Esta capa utiliza función de activación softmax y no utiliza
#Dropout.
#En su reporte de laboratorio incorpore el códigos generado, ası́ como un análisis del número de filtros y
#parámetros de la red final. ¿ Qué capas utilizan más filtros y parámetros ?, ¿Qué justifica este tipo de arqui-
#tectura?. Comente y fundamente sus observaciones.

#Codigo ejecutado
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

#Output
#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayers7-8.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
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
#Capa Densa 7: (None, 4096)
#Capa Densa 8: (None, 1000)
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#zero_padding2d_25 (ZeroPaddi (None, 228, 228, 3)       0         
#_________________________________________________________________
#conv2d_25 (Conv2D)           (None, 55, 55, 96)        34944     
#_________________________________________________________________
#activation_26 (Activation)   (None, 55, 55, 96)        0         
#_________________________________________________________________
#batch_normalization_14 (Batc (None, 55, 55, 96)        384       
#_________________________________________________________________
#max_pooling2d_17 (MaxPooling (None, 27, 27, 96)        0         
#_________________________________________________________________
#zero_padding2d_26 (ZeroPaddi (None, 31, 31, 96)        0         
#_________________________________________________________________
#conv2d_26 (Conv2D)           (None, 27, 27, 256)       614656    
#_________________________________________________________________
#activation_27 (Activation)   (None, 27, 27, 256)       0         
#_________________________________________________________________
#batch_normalization_15 (Batc (None, 27, 27, 256)       1024      
#_________________________________________________________________
#max_pooling2d_18 (MaxPooling (None, 13, 13, 256)       0         
#_________________________________________________________________
#zero_padding2d_27 (ZeroPaddi (None, 15, 15, 256)       0         
#_________________________________________________________________
#conv2d_27 (Conv2D)           (None, 13, 13, 384)       885120    
#_________________________________________________________________
#activation_28 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_28 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_28 (Conv2D)           (None, 13, 13, 384)       1327488   
#_________________________________________________________________
#activation_29 (Activation)   (None, 13, 13, 384)       0         
#_________________________________________________________________
#zero_padding2d_29 (ZeroPaddi (None, 15, 15, 384)       0         
#_________________________________________________________________
#conv2d_29 (Conv2D)           (None, 13, 13, 256)       884992    
#_________________________________________________________________
#activation_30 (Activation)   (None, 13, 13, 256)       0         
#_________________________________________________________________
#max_pooling2d_19 (MaxPooling (None, 6, 6, 256)         0         
#_________________________________________________________________
#flatten_3 (Flatten)          (None, 9216)              0         
#_________________________________________________________________
#dense_2 (Dense)              (None, 4096)              37752832  
#_________________________________________________________________
#activation_31 (Activation)   (None, 4096)              0         
#_________________________________________________________________
#dropout_2 (Dropout)          (None, 4096)              0         
#_________________________________________________________________
#dense_3 (Dense)              (None, 4096)              16781312  
#_________________________________________________________________
#activation_32 (Activation)   (None, 4096)              0         
#_________________________________________________________________
#dropout_3 (Dropout)          (None, 4096)              0         
#_________________________________________________________________
#dense_4 (Dense)              (None, 1000)              4097000   
#_________________________________________________________________
#activation_33 (Activation)   (None, 1000)              0         
#=================================================================
#Total params: 62,379,752
#Trainable params: 62,379,048
#Non-trainable params: 704
#
#Analisis de filtros
#
#Si se analiza la cantidad reportada por summary contra los filtros en bruto de AlexNet estos no coincidirán. Esto es dado a
#los bias vectors en cada capa y al batch normalization. 
#
#Layer	Weights				Raw Parms	Keras Parms 	Bias Vector	Batch Normalization
#1	    96	11	11	3	    34848	    34944	        96	        384
#2	    256	5	5	96	    614400	    614656	        256	        1024
#3	    384	3	3	256	    884736	    885120	        384	
#4	    384	3	3	384	    1327104	    1327488	        384	
#5	    256	3	3	384	    884736	    884992	        256	
#6	    6	6	256	4096	37748736	37752832	    4096	
#7			4096	4096	16777216	16781312	    4096	
#8			4096	1000	4096000	4097000	1000	
#
#Los mayores filtros estan presentes para las capas convulsionales en las capas 4 y 5 y en las capas FC en las capas 6 y 7, que son densas. 
#Asimismo la mayor cantidad de parametros esta presente en las capas convulsionales en la capa 4 y en las capas FC en la capa 6. 
#La justificacion principal de esta arquitectura es la misma que el porque de CNN; reducir una potencial cantidad de conexiones neuronales
#y transformarlas de manera de simplificar el proceso. Algunas observaciones especificas para AlexNet:
#-Batch Normalization permite obtener mayores ratios de aprendizaje
#-La presencia de un bias vector es crucial, ya que permite ajustar la funcion
