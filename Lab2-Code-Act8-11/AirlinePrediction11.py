#Actividad 11
#La red anterior implementa una red convolucional tradicional. Como comentamos en clases, este tipo de
#modelo presenta limitaciones para ajustar los parámetros de la red utilizando métodos de descenso de gra-
#diente (problema de desvanecimiento de gradiente o vanishing gradient problem). En la actualidad redes
#recurrentes tipo LSTM y GRU ofrecen soluciones más estables. Keras ofrece implementaciones de ambas
#alternativas, siendo muy simple su uso, basta cambiar el nombre en el llamado a la función. Por ejemplo,
#para reemplazar en el modelo base la red recurrente tradicional por un modelo LSTM, el código serı́a:

from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Flatten, SimpleRNN,LSTM
import airlineUtils

#set sequence length
history=4 

#read training and test sets
trainX, trainY, testX, testY, scaler, dataset=airlineUtils.readAirlineData(history)

# create and fit the LSTM network
modelRNN = Sequential()
modelRNN.add(LSTM(5,input_shape=(history,1),return_sequences=False))
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

#Output LSTM:
#Train Score: 32.00 RMSE
#Test Score: 75.42 RMSE
#Layer (type)                 Output Shape              Param #   
#=================================================================
#lstm_2 (LSTM)                (None, 5)                 140       
#_________________________________________________________________
#dense_7 (Dense)              (None, 1)                 6         
#=================================================================
#Total params: 146
#Trainable params: 146
#Non-trainable params: 0


#Output RNN para comparacion:
#Train Score: 21.62 RMSE
#Test Score: 47.84 RMSE
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#simple_rnn_6 (SimpleRNN)     (None, 5)                 35        
#_________________________________________________________________
#dense_8 (Dense)              (None, 1)                 6         
#=================================================================
#Total params: 41
#Trainable params: 41
#Non-trainable params: 0

#Como se puede observar, la cantidad de parametros aumenta para LSTM, al igual que el error reportado.   




