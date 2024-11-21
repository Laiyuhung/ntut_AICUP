from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_lstm_model(input_shape):
    """
    Create an LSTM model
    :param input_shape: Shape of the input data (time steps, number of features)
    :return: Created LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))  # Add Dropout to reduce overfitting
    return model

def output_layer_setting(model, output_units):
    """
    Add the output layer and compile the model
    :param model: Created LSTM model
    :param output_units: Number of units in the output layer
    :return: Compiled model
    """
    model.add(Dense(units=output_units))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Compile the model
    return model 

def LSTM_train(X_train, y_train, epochs, batch_size):
    """
    Train the LSTM model and save it
    :param X_train: Training input data (samples, time steps, number of features)
    :param y_train: Training target data (samples, output dimensions)
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    """
    # Check if the input data shape is correct
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1]  # Automatically set output dimensions based on y_train

    # Create the model
    model = create_lstm_model(input_shape)
    model = output_layer_setting(model, output_units)

    # Train the model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Use 20% of the data as validation set
        verbose=1,
        shuffle=True
    )

    # Save the model in .keras format
    model_path = f'./model/LSTM.keras'
    model.save(model_path)
    print(f'Model Saved: {model_path}')

    return model
