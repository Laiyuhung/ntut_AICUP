from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def deep_lstm_model(input_shape):
    model = Sequential()
    
    # 第一層 LSTM
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # 加入 Dropout 層來避免過擬合

    # 第二層 LSTM
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    # 第三層 LSTM
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))

    # 最後的回歸層
    model.add(Dense(units=1))  # 假設是回歸任務，輸出層單位數設為 1

    # 編譯模型
    model.compile(optimizer='adam', loss='mse')  # 使用均方誤差（MSE）作為損失函數

    return model
