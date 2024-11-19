from sklearn.ensemble import VotingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import joblib
import numpy as np
import pandas as pd
import os

def create_modal(AllOutPut, Regression_X_train, Regression_y_train):
    # 對輸入數據進行 Min-Max 正規化
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    
    # 建立基礎回歸模型
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = ExtraTreesRegressor(n_estimators=100, random_state=42)
    
    # 使用 K-Nearest Neighbors Regression 和 XGBoost Regression
    model3 = KNeighborsRegressor(n_neighbors=5)  # n_neighbors 可調整為最佳值
    model4 = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # 使用 VotingRegressor 結合多個模型
    RegressionModel = VotingRegressor(estimators=[
        ('rf', model1),
        ('et', model2),
        ('knn', model3),
        ('xgb', model4)
    ])
    
    # 訓練回歸模型
    RegressionModel.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
    
    return RegressionModel, LSTM_MinMaxModel

def combined_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train):
    # 創建回歸模型
    RegressionModel, LSTM_MinMaxModel = create_modal(AllOutPut, Regression_X_train, Regression_y_train)
    
    # 保存模型
    os.makedirs('./model', exist_ok=True)
    joblib.dump(RegressionModel, f'./model/WeatherRegression_{NowDateTime}')
    
    # 輸出模型的 R^2 分數
    print('Voting Regressor Model R squared: ', RegressionModel.score(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train))