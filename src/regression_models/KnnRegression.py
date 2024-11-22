from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os

def create_modal(AllOutPut, LSTM_MinMaxModel, Regression_X_train, Regression_y_train):
    # Initialize and train the KNN model
    KNNModel = KNeighborsRegressor(n_neighbors=5)
    KNNModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)

    return KNNModel

def knn_regression_modal(NowDateTime, LSTM_MinMaxModel, AllOutPut, Regression_X_train, Regression_y_train):
    # Create the KNN model
    RegressionModel = create_modal(AllOutPut, LSTM_MinMaxModel, Regression_X_train, Regression_y_train)
    
    # Ensure the 'model' directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Save the KNN model using joblib with a .joblib extension
    model_path = f'./models/KNNRegression.joblib'
    joblib.dump(RegressionModel, model_path)

    print(f'Model Saved: {model_path}')
    
    # Output the R squared score
    print('KNN R squared:', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))
