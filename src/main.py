from datetime import datetime
import time

from loading_data import *
from normalize import *
from tranformation_model import *
# from LSTM_model import *
from regression import *
from VotingRegressor import *
from KnnRegression import *
from ExtraTreesRegressor import *
from forcast import *
from forcast_match import *
from status_control import *
from RandomForestRegression import *
from GradientBoostingRegression import *
from SupportVectorRegression import *
from GradientDescentRegression import *

from XgboostRegression import *
from CatboostRegression import *
from LightgbmRegression import *
from ElasticnetRegression import *
from HuberRegression import *
from LassoRegression import *
from RidgeRegression import *

def main():
    start_time = time.time()
    #adding_title( "./data/ExampleTrainData(AVG)" )
    #adding_title( "./data/ExampleTrainData(IncompleteAVG)" )
    SourceData = loading_data( "./data/ExampleTrainData(AVG)" , True)
    AllOutPut = LSTM_data( SourceData ) 
    Regression_X_train , Regression_y_train = regression_data( SourceData )
    
    X_train , y_train , LSTM_MinMaxModel = normal( AllOutPut , 12 )
    # X_train = reshape( x_train )

    regressor = transformer_model((X_train.shape[1], X_train.shape[2]))
    
    NowDateTime = datetime.now().strftime("%Y-%m")


    
    reg_type = ["ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear", "RandomForestRegressor", "GradientBoostingRegressor", "SupportVectorRegressor", "GradientDescentRegression", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Lasso", "Ridge"]
    batch_size_option = [256, 128, 64]
    epoch_option = [50, 100, 150, 200, 250, 300]
    
    # print("aaa")
    for regression_type in reg_type:
        for batch_size in batch_size_option:
            for epochs in epoch_option:
                
                status = check_status(regression_type, batch_size, epochs)
                # print(status)

                if status == 0.0 and regression_type != "GradientDescentRegression":

                    print("--now progressing--")
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
                    total_difference = calculate(regression_type, batch_size, epochs)

                    modify_status(regression_type, batch_size, epochs, total_difference)
                    status_print()



    
    end_time = time.time() 
    execution_time = end_time - start_time
    
    print( f"execution time : {execution_time}") 
main()
