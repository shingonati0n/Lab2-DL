#Actividad 10
#El archivo AirlinePrediction.py, disponible en el sitio web del curso, contiene el cรณdigo anterior y las fun-
#ciones necesarias para cargar el archivo de datos y mostrar los resultados obtenidos. Use este cรณdigo para
#probar el rendimiento del modelo base. En tรฉrminos de la presentaciรณn de resultados, la salida del archivo
#AirlinePrediction.py indica el error medio cuadrรกtico en el set de entrenamiento y test. Ademรกs muestra una
#grรกfica que incluye los datos originales (curva azul), la predicciรณn en datos de entrenamiento (curva verde) y
#la predicciรณn en datos de test (curva roja).
#Ejecute el modelo base varias veces, ยฟPor quรฉ no se obtiene siempre el mismo resultado?. Fundamente su
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
modelRNN.summary()

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

#El resultado varรญa debido a dos factores principales. En primer lugar, a la falta de elementos
#al momento de configurar la red. No hay funcion de activacion, ni tampoco una funcion para inicializar
#el peso. El otro motivo, tiene que ver con los problemas como vanishing gradient y exploding gradient,
#Considerando que el entrenamiento se esta realizando utilizando 150 epochs, la gradiente puede estarse 
#viendo afectada por los cambios en el error. Ya que la red siempre multiplica, en este caso el RSME siempre
#será distinto 


