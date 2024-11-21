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
# from sequence_models.gru_tran import *

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
    # reg_type = ["ExtraTree", "KNNRegression", "VotingRegressor", "Linear", "RandomForest", "GradientBoosting", "SupportVector", "GradientDescent", "XGBoost", "CatBoost", "LightGBM", "ElasticNet", "Huber", "Lasso", "Ridge"]
    # batch_size_option = [256, 128, 64]
    # epoch_option = [50, 100, 150, 200, 250, 300]

    # hopes
    seq_type = "BidirectionalLSTM" 
    reg_type = "ExtraTree" 
    batch_size = 128
    epochs = 200

    print("---now progressing---")
    print("Sequencial model type: ", seq_type)
    print("Regression type: ", reg_type)
    print("Batch size: ", batch_size)
    print("Epochs: ", epochs)
    print()

    if seq_type == "Transformer":
        transformer_train(X_train, y_train, epochs, batch_size)

    elif seq_type == "LSTM":
        LSTM_train(X_train, y_train, epochs, batch_size)
    
    elif seq_type == "GRU":
        gru_train(X_train, y_train, epochs, batch_size)
    
    elif seq_type == "SimpleRNN":
        simple_rnn_train(X_train, y_train, epochs, batch_size)

    elif seq_type == "BidirectionalLSTM":
        bidirectional_lstm_train(X_train, y_train, epochs, batch_size)




    Regression_y_train = Regression_y_train.ravel()
    if reg_type == "ExtraTree":
        ExtraTree_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

    elif reg_type == "KNNRegression":
        knn_regression_modal( NowDateTime , LSTM_MinMaxModel, AllOutPut , Regression_X_train , Regression_y_train )
    
    # elif reg_type == "Voting":
    #     voting_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )

    elif reg_type == "Linear":
        linear_regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )
    
    elif reg_type == "RandomForest":
        random_forest_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

    elif reg_type == "GradientBoosting":
        gradient_boosting_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

    elif reg_type == "SupportVector":
        support_vector_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)
    
    elif reg_type == "XGBoost":
        xgboost_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

    elif reg_type == "CatBoost":
        catboost_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

    elif reg_type == "LightGBM":
        lightgbm_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

    elif reg_type == "ElasticNet":
        elasticnet_regression_modal(NowDateTime, Regression_X_train, Regression_y_train)

    # elif reg_type == "Huber":
    #     huber_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

    # elif reg_type == "Lasso":
    #     lasso_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

    # elif reg_type == "Ridge":
    #     ridge_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

    # elif reg_type == "GradientDescentRegression":
    #     gradient_descent_regression_modal(NowDateTime, AllOutPut, Regression_X_train, Regression_y_train)

    lstm_model_name = seq_type
    regression_model_name = reg_type
    print("lstm_model: ",lstm_model_name)
    print("regression_model: ",regression_model_name)
    forcast( AllOutPut = AllOutPut , lstm = f'./model/{lstm_model_name}.keras' , regression_model = f'./models/{regression_model_name}Regression.joblib',k=1 )


    end_time = time.time() 
    execution_time = end_time - start_time
    
    print( f"execution time : {execution_time}") 
main()
