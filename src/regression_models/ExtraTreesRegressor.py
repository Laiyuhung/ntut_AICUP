from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
import os

def create_modal(AllOutPut, Regression_X_train, Regression_y_train):
    # Fit MinMaxScaler on the dataset
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    
    # Initialize and train the ExtraTreesRegressor model
    RegressionModel = ExtraTreesRegressor(n_estimators=100, random_state=42)
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    
    return RegressionModel, LSTM_MinMaxModel

def ExtraTree_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train):
    # Create model and MinMaxScaler
    RegressionModel, LSTM_MinMaxModel = create_modal(AllOutPut, Regression_X_train, Regression_y_train)
    
    # Ensure the 'model' directory exists
    os.makedirs('./model', exist_ok=True)
    
    # Save the ExtraTreesRegressor model using joblib with a .joblib extension
    model_path = f'./models/ExtraTreeRegression.joblib'
    joblib.dump(RegressionModel, model_path)
    
    # Save the MinMaxScaler as well
    scaler_path = './models/ExtraTreeMinMaxScaler.joblib'
    joblib.dump(LSTM_MinMaxModel, scaler_path)
    
    print(f'Model Saved: {model_path}')
    print(f'Scaler Saved: {scaler_path}')
    
    # Print feature importance and R-squared score
    print('Feature Importances: ', RegressionModel.feature_importances_)
    print('R squared: ', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))
