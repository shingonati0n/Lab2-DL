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
modelRNN.add(Dense(1))
modelRNN.add(Dropout(0.2))

#Train model
modelRNN.compile(loss='mean_squared_error', optimizer='adam')
modelRNN.fit(trainX, trainY, epochs=150, batch_size=5, verbose=2)

# Make predictions
trainPredict = modelRNN.predict(trainX)
testPredict = modelRNN.predict(testX)

#Display results
airlineUtils.displayResult(dataset, trainPredict, trainY, testPredict, testY, scaler, history)

#Output:

#Train Score: 20.34 RMSE
#Test Score: 55.89 RMSE

#Es posible observar en el grafico que las curvas de training y testing cambian; al cambiar el valor de
#history, se está tomando la informacion de los ultimos 10 meses. Se puede ver que los primeros10 epochs vs 
#el dataset original no son considerados; algo que se repite en el testing, donde solo predice despues del
#10mo Epoch.

#• Agregue una nueva capa densa de 100 unidades y vea su impacto en los resultados.

#Output:

#ValueError: Error when checking target: expected dense_10 to have shape (None, 100) but got array with shape (85, 1)

#Al intentar agregar una capa densa de 100 unidades, el programa falla. Esto ocurre por diseño: 
#el momento en que se ajusta el modelo (fit), el programa espera que la entrada tenga un shape con 
#1 clase, sin embargo, al aplicar Dense(100), se provoca el error descrito.
#Una evidencia de esto es el output con Dense(1) y obteniendo el summary de la ejecución:

#Layer (type)                 Output Shape              Param #   
#=================================================================
#simple_rnn_14 (SimpleRNN)    (None, 5)                 35        
#_________________________________________________________________
#dense_11 (Dense)             (None, 1)                 6         
#=================================================================

#El programa deberia tener 100 clases para que una insercion de capa densa de 100 unidades funcionase.

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

#2014 paper Dropout: A Simple Way to Prevent Neural Networks from Overfitting (download the PDF).