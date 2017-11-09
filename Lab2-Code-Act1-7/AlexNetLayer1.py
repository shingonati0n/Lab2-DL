#Actividad 1

#El archivo AlexNetCapa1.py, disponible en el sitio web del curso, contiene el código anterior. Ejecute este
#código y verifique que las dimensiones de salida de la capa definida corresponden a las utilizadas por AlexNet.
#Para esto puede utilizar en Keras el comando: print(modelAlexNet.output shape). Este comando permite
#imprimir en pantalla la dimensiones de salida de la red definida hasta la ejecución del comando. A modo
#de ejemplo, para acceder a las dimensiones de salida de la red antes y después de aplicar el operador Max-
#Pooling, podemos ejecutar:

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

#Output:
#runfile('/home/F5/Documents/ML-DL/Laboratorio2/Generated Code/AlexNetLayer1.py', wdir='/home/F5/Documents/ML-DL/Laboratorio2/Generated Code')
#Using TensorFlow backend.
#/usr/lib64/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
#  return f(*args, **kwds)
#Batch Normalization Capa 1: (None, 55, 55, 96) 
#Max Pooling Capa 1: (None, 27, 27, 96)
#las dimensiones corresponden a las de la segunda capa de AlexNet. Cabe destacar que se cambio algo del codigo respecto a lo entregado para este laboratorio. 
#Dado a que estoy utilizando Python 3.6 y Keras 2 API, estoy utilizando Conv2D en vez de Convulsion2D, Adicionalmente, agregue una descripción a cada output para guiar mejor
#la lectura del mismo. Estos seran utilizados en el resto del codigo


