#Actividad 10
#Luego pruebe las siguientes variantes al modelo base y analice su resultados:
#• Modifique el largo de la secuencia de entrada. Este valor es determinado por el valor de la variable
#history. Por ejemplo, para cambiar el largo de la secuencia de entrada desde 4 (modelo base) a 10, debe
#reemplazar en AirlinePrediction.py history=4 por history=10.

#Ejecucion con history=10

from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Flatten, SimpleRNN,LSTM
import airlineUtils

#set sequence length
history=10

#read training and test sets
trainX, trainY, testX, testY, scaler, dataset=airlineUtils.readAirlineData(history)

# create and fit the LSTM network
modelRNN = Sequential()
modelRNN.add(SimpleRNN(5,input_shape=(history,1),return_sequences=False))
modelRNN.add(Dense(100))
modelRNN.add(Dropout(0.5))
modelRNN.add(Dense(1))


#Train model
modelRNN.compile(loss='mean_squared_error', optimizer='adam')
print(modelRNN.output_shape)
modelRNN.fit(trainX, trainY, epochs=150, batch_size=5, verbose=2)
print(modelRNN.output_shape)

# Make predictions
trainPredict = modelRNN.predict(trainX)
testPredict = modelRNN.predict(testX)

#Display results
airlineUtils.displayResult(dataset, trainPredict, trainY, testPredict, testY, scaler, history)
modelRNN.summary()

#Output:

#Train Score: 20.34 RMSE
#Test Score: 55.89 RMSE

#Es posible observar en el grafico que las curvas de training y testing cambian; al cambiar el valor de
#history, se está tomando la informacion de los ultimos 10 meses. Se puede ver que los primeros10 epochs vs 
#el dataset original no son considerados; algo que se repite en el testing, donde solo predice despues del
#10mo Epoch.

#• Agregue una nueva capa densa de 100 unidades y vea su impacto en los resultados.

#Output:

#Train Score: 22.30 RMSE
#Test Score: 79.22 RMSE

#Al agregar una capa densa de 100 unidades, se generan parametros adicionales. Esto afecta en que la red  
#toma un poco mas de tiempo para entrenar, dado a estos parametros adiconales.

#Layer (type)                 Output Shape              Param #   
#=================================================================
#simple_rnn_8 (SimpleRNN)     (None, 5)                 35        
#_________________________________________________________________
#dense_12 (Dense)             (None, 100)               600       
#_________________________________________________________________
#dense_13 (Dense)             (None, 1)                 101       
#=================================================================
#Total params: 736
#Trainable params: 736
#Non-trainable params: 0

#• Agregue Dropout.

#Codigo Ejecutado:

#modelRNN = Sequential()
#modelRNN.add(SimpleRNN(5,input_shape=(history,1),return_sequences=False))
#modelRNN.add(Dense(1))
#modelRNN.add(Dropout(0.2))

#Output:
#Train Score: 34.32 RMSE
#Test Score: 80.66 RMSE

#Se insertó un dropout de 20% inicialmente, segun lo recomendado por Srivastava, et al.
#Como se puede apreciar, dropout ocasiona que exista underfitting, dado a que se estan ignorando un 
#20% de neuronas. Para este caso particular, la medida que resultó mas optima en Dropout fue de un 5%:

#Output:
#Train Score: 21.67 RMSE
#Test Score: 49.81 RMSE

#Sin embargo, si se posiciona el Dropout entre Dense(100) y Dense(1), esto puede llevar a un entrenamiento
#mas preciso; Al existir mas parametros y remover varios de ellos, se equilibra el training:

#Codigo ejecutado con Dropout + Dense

#modelRNN = Sequential()
#modelRNN.add(SimpleRNN(5,input_shape=(history,1),return_sequences=False))
#modelRNN.add(Dense(100))
#modelRNN.add(Dropout(0.5))
#modelRNN.add(Dense(1))

#Output:

#Train Score: 18.88 RMSE
#Test Score: 52.64 RMSE

#Layer (type)                 Output Shape              Param #   
#=================================================================
#simple_rnn_9 (SimpleRNN)     (None, 5)                 35        
#_________________________________________________________________
#dense_14 (Dense)             (None, 100)               600       
#_________________________________________________________________
#dropout_7 (Dropout)          (None, 100)               0         
#_________________________________________________________________
#dense_15 (Dense)             (None, 1)                 101       
#=================================================================
#Total params: 736
#Trainable params: 736
#Non-trainable params: 0

#2014 paper Dropout: A Simple Way to Prevent Neural Networks from Overfitting (download the PDF).