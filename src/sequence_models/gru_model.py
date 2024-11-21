import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling1D
from datetime import timedelta

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
import os

def gru_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # GRU layer for sequence processing
    gru_output = GRU(128, return_sequences=True)(inputs)
    gru_output = LayerNormalization(epsilon=1e-6)(gru_output)

    # Add another GRU layer for more depth
    gru_output = GRU(64, return_sequences=True)(gru_output)
    gru_output = LayerNormalization(epsilon=1e-6)(gru_output)

    # Global pooling layer to reduce dimensions
    gru_output = GlobalAveragePooling1D()(gru_output)
    
    # Final dense layer for output
    outputs = Dense(10)(gru_output)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

def gru_train(X_train, y_train, epochs=100, batch_size=64):
    # Initialize the GRU model
    regressor = gru_model((X_train.shape[1], X_train.shape[2]))
    
    # Train the GRU model
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the trained model
    NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    regressor.save('./model/GRU.keras')
    print('Model Saved: GRU.keras')
