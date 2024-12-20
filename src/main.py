from datetime import datetime
import time
import sys


from loading_data import *
from normalize import *
from forcast import *
from forcast_match import *
from status_control import *
from competition import *

sys.path.append('/sequence_models/')
from sequence_models.tranformation_model import *
from sequence_models.LSTM_model import *
from sequence_models.gru_model import *
from sequence_models.simple_rnn_model import *
from sequence_models.bidirectional_LSTM import *
from sequence_models.martin import *

sys.path.append('/regression_models/')
from regression_models.Linear import *
from regression_models.VotingRegressor import *
from regression_models.KnnRegression import *
from regression_models.ExtraTreesRegressor import *
from regression_models.RandomForestRegression import *
from regression_models.GradientBoostingRegression import *
from regression_models.SupportVectorRegression import *
# from regression_models.GradientDescentRegression import *
from regression_models.XgboostRegression import *
from regression_models.CatboostRegression import *
from regression_models.LightgbmRegression import *
from regression_models.ElasticnetRegression import *
from regression_models.martinRegression import *
# from regression_models.HuberRegression import *
# from regression_models.LassoRegression import *
# from regression_models.RidgeRegression import *

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

    # orgiginal
    # seq_type = ["Transformer", "LSTM", "GRU", "SimpleRNN", "BidirectionalLSTM"]
    # reg_type = ["ExtraTree", "KNN", "Voting", "Linear", "RandomForest", "GradientBoosting", "SupportVector", "GradientDescent", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Lasso", "Ridge"]
    # batch_size_option = [256, 128, 64]
    # epoch_option = [50, 100, 150, 200, 250, 300]

    # hopes
    seq_type = ["Combined", "LSTM"]
    reg_type = ["KNN", "Voting", "Linear", "CatBoost", "LightGBM", "ElasticNet"]
    batch_size_option = [256]
    epoch_option = [100, 200]
    start_k = 166
    # seq_type = ["Transformer"]
    # reg_type = ["Linear", "GradientBoosting"]
    # batch_size_option = [256]
    # epoch_option = [100, 150]

    

    for sequencial in seq_type:
        for regression in reg_type:
            for batch in batch_size_option:
                for epoch in epoch_option:

                    print("---now progressing---")
                    print("Sequencial model type: ", sequencial)
                    print("Regression type: ", regression)
                    print("Batch size: ", batch)
                    print("Epochs: ", epoch)
                    print()

                    if sequencial == "Transformer":
                        transformer_train(X_train, y_train, epoch, batch)

                    elif sequencial == "LSTM":
                        LSTM_train(X_train, y_train, epoch, batch)
                    
                    elif sequencial == "GRU":
                        gru_train(X_train, y_train, epoch, batch)
                    
                    elif sequencial == "SimpleRNN":
                        simple_rnn_train(X_train, y_train, epoch, batch)

                    elif sequencial == "BidirectionalLSTM":
                        bidirectional_lstm_train(X_train, y_train, epoch, batch)

                    elif sequencial == "Combined":
                        combined_train(X_train, y_train, epoch, batch)



                    Regression_y_train = Regression_y_train.ravel()
                    if regression == "ExtraTree":
                        ExtraTree_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                    elif regression == "KNN":
                        knn_regression_modal( NowDateTime , LSTM_MinMaxModel, AllOutPut , Regression_X_train , Regression_y_train )
                    
                    elif regression == "Voting":
                        voting_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                    elif regression == "Linear":
                        linear_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )
                    
                    elif regression == "RandomForest":
                        random_forest_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

                    elif regression == "GradientBoosting":
                        gradient_boosting_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

                    elif regression == "SupportVector":
                        support_vector_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)
                    
                    elif regression == "XGBoost":
                        xgboost_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

                    elif regression == "CatBoost":
                        catboost_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

                    elif regression == "LightGBM":
                        lightgbm_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

                    elif regression == "ElasticNet":
                        elasticnet_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

                    elif regression == "Combined":
                        combined_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

                    # elif regression == "Huber":
                    #     huber_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                    # elif regression == "Lasso":
                    #     lasso_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                    # elif regression == "Ridge":
                    #     ridge_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                    # elif regression == "GradientDescent":
                    #     gradient_descent_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

                    lstm_model_name = sequencial
                    regression_model_name = regression
                    print("lstm_model: ",lstm_model_name)
                    print("regression_model: ",regression_model_name)
                    print("file No. : ",start_k)
                    forcast( AllOutPut = AllOutPut , lstm = f'./model/{lstm_model_name}.keras' , regression_model = f'./models/{regression_model_name}Regression.joblib', k=start_k, sequencial=sequencial, regression=regression, batch=batch, epoch=epoch )
                    start_k = start_k+1

    end_time = time.time() 
    execution_time = end_time - start_time
    
    print( f"execution time : {execution_time}") 
main()
