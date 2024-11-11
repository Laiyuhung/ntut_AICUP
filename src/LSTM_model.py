from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

def create_lstm_model( X_train ):
    regressor = Sequential ()
    
    regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], 2)))
    regressor.add(LSTM(units =  64))
    regressor.add(Dropout(0.2))
    
    return regressor

def output_layer_setting( regressor ):
    regressor.add(Dense(units = 2))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor

def train( X_train , y_train , NowDateTime , epochs , batch_size):
    regressor = create_lstm_model( X_train )
    regressor = output_layer_setting( regressor )
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
    
    from datetime import datetime
    regressor.save('./model/WheatherLSTM_' + NowDateTime+'.h5')
    print('Model Saved')

