from datetime import datetime
import time
import sys


from loading_data import *
from normalize import *
from forcast import *
from forcast_match import *
from status_control import *

sys.path.append('/sequence_models/')
from sequence_models.tranformation_model import *
from sequence_models.LSTM_model import *
from sequence_models.gru_model import *
from sequence_models.simple_rnn_model import *
from sequence_models.bidirectional_LSTM import *


sys.path.append('/regression_models/')
from regression import *
from regression_models.VotingRegressor import *
from regression_models.KnnRegression import *
from regression_models.ExtraTreesRegressor import *
from regression_models.RandomForestRegression import *
from regression_models.GradientBoostingRegression import *
from regression_models.SupportVectorRegression import *
from regression_models.GradientDescentRegression import *
from regression_models.XgboostRegression import *
from regression_models.CatboostRegression import *
from regression_models.LightgbmRegression import *
from regression_models.ElasticnetRegression import *
from regression_models.HuberRegression import *
from regression_models.LassoRegression import *
from regression_models.RidgeRegression import *

def main():
    start_time = time.time()
    #adding_title( "./data/ExampleTrainData(AVG)" )
    #adding_title( "./data/ExampleTrainData(IncompleteAVG)" )
    SourceData = loading_data( "./data/ExampleTrainData(AVG)" , True)
    AllOutPut = LSTM_data( SourceData ) 
    Regression_X_train , Regression_y_train = regression_data( SourceData )
    
    X_train , y_train , LSTM_MinMaxModel = normal( AllOutPut , 12 )
    # X_train = reshape( x_train )

    
    
    
        
    
    
    NowDateTime = datetime.now().strftime("%Y-%m")

    #orgiginal
    # seq_type = ["Transformer", "LSTM", "GRU", "Simple RNN", "Bidirectional LSTM"]
    # reg_type = ["ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "GradientDescentRegression", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Lasso", "Ridge"]
    # batch_size_option = [256, 128, 64]
    # epoch_option = [50, 100, 150, 200, 250, 300]

    #hopes
    # seq_type = ["Transformer", "GRU", "Bidirectional LSTM", "LSTM", "Simple RNN"]
    seq_type = ["LSTM"]
    # reg_type = ["Lasso", "ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Ridge"]
    reg_type = ["Linear"]
    batch_size_option = [128]
    epoch_option = [100]

    for sequential_type in seq_type:

        if sequential_type == "Transformer":
            regressor = transformer_model((X_train.shape[1], X_train.shape[2]))

        elif sequential_type == "LSTM":
            regressor = deep_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        if sequential_type == "GRU":
            regressor = gru_model((X_train.shape[1], X_train.shape[2]))
        
        elif sequential_type == "Simple RNN":
            regressor = simple_rnn_model((X_train.shape[1], X_train.shape[2]))

        elif sequential_type == "Bidirectional LSTM":
            regressor = bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]))

        # print(now_seq)
        for regression_type in reg_type:
            for batch_size in batch_size_option:
                for epochs in epoch_option:
                    
                    # status = check_status(sequential_type, regression_type, batch_size, epochs)
                    # print(status)

                    if status == 0.0 :

                        print("--now progressing--")
                        print("sequencial_model_type: ", sequential_type)
                        print("regression_type: ", regression_type)
                        print("batch_size: ", batch_size)
                        print("epochs: ", epochs)
                        train( X_train, y_train, epochs, batch_size)
                        
                        if regression_type == "ExtraTreesRegressor":
                            ExtraTree_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                        elif regression_type == "KnnRegression":
                            knn_regression_modal( NowDateTime , LSTM_MinMaxModel, AllOutPut , Regression_X_train , Regression_y_train )
                        
                        elif regression_type == "VotingRegressor":
                            voting_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                        elif regression_type == "Linear":
                            regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )
                        
                        elif regression_type == "RandomForestRegressor":
                            random_forest_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "GradientBoostingRegressor":
                            gradient_boosting_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "SupportVectorRegressor":
                            support_vector_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)
                        
                        

                        #new
                        elif regression_type == "XGBoost":
                            xgboost_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "CatBoost":
                            catboost_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "LightGBM":
                            lightgbm_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "ElasticNet":
                            elasticnet_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "Huber":
                            huber_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "Lasso":
                            lasso_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "Ridge":
                            ridge_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                        elif regression_type == "GradientDescentRegression":
                            gradient_descent_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)


                        forcast( AllOutPut = AllOutPut , lstm = 'WeatherTransformer.keras' , regression_model = f'./model/WeatherRegression_{NowDateTime}' )
                        total_difference = calculate(sequential_type, regression_type, batch_size, epochs)

                        modify_status(sequential_type, regression_type, batch_size, epochs, total_difference)
                        status_print()
                        sort_result()



    
    end_time = time.time() 
    execution_time = end_time - start_time
    
    print( f"execution time : {execution_time}") 
main()
