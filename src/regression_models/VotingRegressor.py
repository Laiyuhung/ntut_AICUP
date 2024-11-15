from sklearn.ensemble import VotingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
import os

def create_modal(AllOutPut, Regression_X_train, Regression_y_train):
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model3 = LinearRegression()

    RegressionModel = VotingRegressor(estimators=[
        ('rf', model1),
        ('et', model2),
        ('lr', model3)
    ])
    
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    
    return RegressionModel, LSTM_MinMaxModel

def voting_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train):
    RegressionModel, LSTM_MinMaxModel = create_modal(AllOutPut, Regression_X_train, Regression_y_train)
    
    os.makedirs('./model', exist_ok=True)
    joblib.dump(RegressionModel, f'./model/WeatherRegression_{NowDateTime}')
    
    print('Voting Regressor Model R squared: ', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))
