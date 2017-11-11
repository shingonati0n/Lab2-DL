#Actividad 12:

#Considere la función: modelRNN.fit(trainX, trainY, nb epoch=150, batch size=5, verbose=2)
#Explique brevemente el rol de los parámetros: nb epoch y batch size. Seleccione alguno de los modelos
#testeados y ejecute cambiando el valor de estos 2 parámetros, comente sus observaciones.

#De acuerdo con la documentacion oficial de Keras:

#batch_size: Es el numero de muestras por cada actualización de gradiente. Se representa por un numero entero.

#Respecto de nb_epoch, aca se explicara el parametro epochs, ya que nb_epoch sera deprecado
#epochs: Es el numero de pasos de tiempo bajo el cual entrenará un modelo. Si se utiliza en conjunto con el parametro initial_epoch, 
#su significado cambia y pasaria a ser el ultimo epoch, vale decir, el modelo es entrenado no por el numero de pasos dado por epochs, 
#sino hasta que el epoch epochs haya sido alcanzado.

#Ejecucion 1: ejecucion inicial SimpleRNN cambiando batch_size a 4 y epochs a 50:

from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Flatten, SimpleRNN,LSTM
import airlineUtils

#set sequence length
history=4

#read training and test sets
trainX, trainY, testX, testY, scaler, dataset=airlineUtils.readAirlineData(history)

# create and fit the LSTM network
modelRNN = Sequential()
modelRNN.add(SimpleRNN(5,input_shape=(history,1),return_sequences=False))
modelRNN.add(Dense(1))

#Train model
modelRNN.compile(loss='mean_squared_error', optimizer='adam')
modelRNN.fit(trainX, trainY, epochs=50, batch_size=2, verbose=2)

# Make predictions
trainPredict = modelRNN.predict(trainX)
testPredict = modelRNN.predict(testX)

#Display results
airlineUtils.displayResult(dataset, trainPredict, trainY, testPredict, testY, scaler, history)
modelRNN.summary()

#Output:
#Train Score: 22.50 RMSE
#Test Score: 47.73 RMSE

#Ejecucion 2: ejecucion de LSTM con epochs=50 y batch_size=2
#Output:
#Train Score: 29.42 RMSE
#Test Score: 68.55 RMSE

#Es posible apreciar que al ejecutar con un valor menor de epochs, el entrenamiento se vuelve mas rapido, dado a que existen menos pasos. 
#El Error medio se mantiene similar en ambos casos. Cabe destacar que en el caso de SimpleRNN, al disminuir el tamaño de batch, muestra signos de 
#overfitting. En LSTM, es al contrario, el model de training no se ajusta a los datos de training, dando un modelo con underfitting.