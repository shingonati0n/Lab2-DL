from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Flatten, SimpleRNN,LSTM
import airlineUtils

#set sequence length
history=4 

#Actividad 8
#Verifique que el modelo definido anteriormente contiene 35 parámetros. Para ello utilice como guı́a el código
#anterior y la función summary() para acceder al número de parámetros.

# Codigo ejecutado
# create and fit the LSTM network
modelRNN = Sequential()
#modelRNN.add(SimpleRNN(5,input_dim=1,input_length=history, return_sequences=False))
modelRNN.add(SimpleRNN(5,input_shape=(history, 1), return_sequences=False))
modelRNN.summary()

#Cabe destacar que se esta utilizando input_shape en vez de input_dim y input_length, por recomendacion de
#Keras 2 API
#output
#
#runfile('/home/F5/Lab2-DL/Lab2-Code-Act8-11/AirlinePrediction8.py', wdir='/home/F5/Lab2-DL/Lab2-Code-Act8-11')
#Reloaded modules: airlineUtils
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#simple_rnn_2 (SimpleRNN)     (None, 5)                 35        
#=================================================================
#Total params: 35
#Trainable params: 35
#Non-trainable params: 0
#
#Si se decide modificar la dimensionalidad del estado intermedio a un valor 4, ¿Cómo se afecta el número de
#parámetros?. Pruebe y fundamente su respuesta.
#
# Codigo ejecutado
# create and fit the LSTM network
modelRNN2 = Sequential()
modelRNN2.add(SimpleRNN(4,input_shape=(history, 1), return_sequences=False))
modelRNN2.summary()

#output:
#Layer (type)                 Output Shape              Param #   
#=================================================================
#simple_rnn_4 (SimpleRNN)     (None, 4)                 24        
#=================================================================
#Total params: 24
#Trainable params: 24
#Non-trainable params: 0
#
#Al cambiar el valor de la dimensionalidad intermedia a 4, el total de parametros baja a 24. Esto
#se explica porque al bajar la dimensionalidad, en este caso iria de 1 a 4, por lo cual el embedding 
#Wxh ∈ R^4x1 daría 4 parámetros. Por su parte, para el embedding entre estados intermedios sería 
#Whh ∈ R 4x4, lo que da, 16 parámetros. Finalmente, el embedding de salida Why ∈ R 4x1 que son 4
# parámetros, lo que da un total de parametros de 24. 