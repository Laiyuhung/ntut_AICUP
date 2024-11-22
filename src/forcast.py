import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

def forcast(AllOutPut, lstm, regression_model, k=0, sequencial="test", regression="test", batch=128, epoch=100):
    # Initialize the MinMaxScaler with AllOutPut
    LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
    
    # Load the LSTM model (.h5 or .keras)
    regressor = load_model(lstm)
    
    # Load the regression model
    Regression = joblib.load(regression_model)
    
    # Parameters
    LookBackNum = 12
    ForecastNum = 48
    
    # Load input data
    data_name = './data/ExampleTestData/upload(noanswer).csv'
    source_data = pd.read_csv(data_name, encoding='utf-8')
    target = ['序號']
    ex_question = source_data[target].values
    inputs = []
    predict_output = []
    predict_power = []
    count = 0

    # Start forecasting
    while count < len(ex_question):
        print('count : ', count)
        LocationCode = int(ex_question[count])
        strLocationCode = str(LocationCode)[-2:]
        if LocationCode < 10:
            strLocationCode = '0' + str(LocationCode)

        DataName = f'./data/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_{strLocationCode}_modified3.csv'
        SourceData = pd.read_csv(DataName, encoding='utf-8')

        # Use the extended fields
        ReferTitle = SourceData[['Serial']].values
        ReferData = SourceData[
            ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)',
             'Hour', 'Season_weight', 'Sunlight_time(h)', 'UV', 'Cloud']
        ].values

        inputs = []

        for DaysCount in range(len(ReferTitle)):
            if str(int(ReferTitle[DaysCount]))[:8] == str(int(ex_question[count]))[:8]:
                TempData = ReferData[DaysCount].reshape(1, -1)
                TempData = LSTM_MinMaxModel.transform(TempData)
                inputs.append(TempData)

        for i in range(ForecastNum):
            if i > 0:
                inputs.append(predict_output[i - 1].reshape(1, 10))  # Expand to 10 columns

            X_test = []
            X_test.append(inputs[0 + i:LookBackNum + i])

            NewTest = np.array(X_test)
            NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 10))  # Adjust input dimension

            # Predict using LSTM model
            predicted = regressor.predict(NewTest)

            # Ensure the shape is correct
            if predicted.ndim == 1:
                predicted = predicted.reshape(-1, 1)  # Reshape to (n_samples, 1)
            if predicted.shape[1] != 10:
                predicted = np.tile(predicted, (1, 10))  # Expand to 10 columns

            predict_output.append(predicted)

            # Further prediction using regression model
            regression_prediction = Regression.predict(predicted)
            predict_power.append(np.round(regression_prediction, 5).flatten())

        count += 48

    # Create DataFrame for results
    df = pd.DataFrame(predict_power, columns=['答案'])
    df.insert(0, '序號', ex_question)
    df.to_csv(f'./result/{k}_output_{sequencial}_{regression}_{batch}_{epoch}.csv', index=False)
    print('Output CSV File Saved')
