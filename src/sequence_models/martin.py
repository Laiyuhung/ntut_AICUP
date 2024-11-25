from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Input, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 创建 LSTM + CNN + Transformer 模型
def create_lstm_cnn_transformer_model(X_train):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))  # 支持 3 个特征的输入

    # CNN 层：用于提取局部特征
    cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn_layer)

    # LSTM 层：用于提取时间序列特征
    lstm_layer = LSTM(units=128, return_sequences=True)(cnn_layer)
    lstm_layer = LSTM(units=64, return_sequences=True)(lstm_layer)
    lstm_layer = Dropout(0.2)(lstm_layer)

    # Transformer 层：增强建模能力
    transformer_layer = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_layer, lstm_layer)
    transformer_layer = Add()([lstm_layer, transformer_layer])  # 残差连接
    transformer_layer = LayerNormalization()(transformer_layer)

    # 输出层
    output = Flatten()(transformer_layer)
    output = Dense(units=3)(output)  # 输出 3 个预测值（对应特征）
    
    # 构建模型
    model = Model(inputs=input_layer, outputs=output)
    return model

# 编译和设置输出层
def output_layer_setting(regressor):
    regressor.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return regressor

# 训练模型
def combined_train(X_train, y_train, NowDateTime, epochs, batch_size):
    regressor = create_lstm_cnn_transformer_model(X_train)
    regressor = output_layer_setting(regressor)
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    regressor.save('./model/Combined.keras')  # Changed the file extension to .keras
    print('Model Saved')