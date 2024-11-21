import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN, LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling1D
from datetime import datetime

import numpy as np
import pandas as pd
import os

def simple_rnn_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Simple RNN layer for sequence processing
    rnn_output = SimpleRNN(128, return_sequences=True)(inputs)
    rnn_output = LayerNormalization(epsilon=1e-6)(rnn_output)

    # Add another RNN layer for more depth
    rnn_output = SimpleRNN(64, return_sequences=True)(rnn_output)
    rnn_output = LayerNormalization(epsilon=1e-6)(rnn_output)

    # Global pooling layer to reduce dimensions
    rnn_output = GlobalAveragePooling1D()(rnn_output)
    
    # Final dense layer for output
    outputs = Dense(10)(rnn_output)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

def simple_rnn_train(X_train, y_train, epochs=100, batch_size=64):
    # Initialize the Simple RNN model
    regressor = simple_rnn_model((X_train.shape[1], X_train.shape[2]))
    
    # Train the RNN model
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the trained model
    NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    regressor.save('./model/SimpleRNN.keras')
    print('Model Saved: SimpleRNN.keras')
