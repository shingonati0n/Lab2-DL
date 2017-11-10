#Actividad 10
#El archivo AirlinePrediction.py, disponible en el sitio web del curso, contiene el código anterior y las fun-
#ciones necesarias para cargar el archivo de datos y mostrar los resultados obtenidos. Use este código para
#probar el rendimiento del modelo base. En términos de la presentación de resultados, la salida del archivo
#AirlinePrediction.py indica el error medio cuadrático en el set de entrenamiento y test. Además muestra una
#gráfica que incluye los datos originales (curva azul), la predicción en datos de entrenamiento (curva verde) y
#la predicción en datos de test (curva roja).
#Ejecute el modelo base varias veces, ¿Por qué no se obtiene siempre el mismo resultado?. Fundamente su
#respuesta.

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
modelRNN.fit(trainX, trainY, epochs=150, batch_size=5, verbose=2)

# Make predictions
trainPredict = modelRNN.predict(trainX)
testPredict = modelRNN.predict(testX)

#Display results
airlineUtils.displayResult(dataset, trainPredict, trainY, testPredict, testY, scaler, history)

#Cabe destacar que al igual que con los ejercicios anteriores, el codigo ha sido levemente actualizado.

#Output 1:
#Train Score: 22.54 RMSE
#Test Score: 49.90 RMSE 

#Output 2:
#Train Score: 21.69 RMSE
#Test Score: 54.57 RMSE

#Output 3:
#Train Score: 24.23 RMSE
#Test Score: 56.35 RMSE

#Output 4:
#Train Score: 25.77 RMSE
#Test Score: 77.36 RMSE

#El resultado varía debido a que se presentan problemas de desvanecimiento de gradiente. Esto
#es evidente mirando los graficos de cada ejecucion. 