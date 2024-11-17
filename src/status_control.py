import itertools
import numpy as np
import pandas as pd
from itertools import product


def build_status(seq_model_type, reg_type, batch_size_option, epoch_option):

    loaded_array = np.load('./result/progress_status.npy')
    
    cartesian = np.array(list(product([seq_model_type],[reg_type], batch_size_option, epoch_option)))
    status = 0

    print(cartesian)
    # new_array = []
    # for item in loaded_array:
    #     # 將插入後的結果加入新陣列
    #     new_item = np.insert(item, 0, seq_model_type)
    #     new_array.append(new_item)
    # new_array = np.array(new_array)
    # print(new_array)

    for item in cartesian:
        cartesian_with_status = np.array([item[0], item[1], item[2],  item[3], status])
        cartesian_with_status = cartesian_with_status.reshape(1, -1)
        # print(cartesian_with_status)
        loaded_array = np.append(loaded_array, cartesian_with_status, axis=0)
    

    print(loaded_array)
    # print(cartesian_with_status)

    # loaded_array = np.append(loaded_array, cartesian_with_status, axis=0)
    # print(loaded_array)
    
    # loaded_array = np.delete(loaded_array, np.arange(-6, 0), axis=0)

    # print(loaded_array)


    # np.save('./result/progress_status.npy', loaded_array)
    # result_to_csv()


def status_print():

    loaded_array = np.load('./result/progress_status.npy')
    print(loaded_array)

    min_error = 10000000
    
    for item in loaded_array:
        if float(item[4]) < min_error and float(item[4])!=0:
            min_combination = item
            min_error = float(item[4])

    print("Best: ",min_combination)

def manuel_modify_status():

    loaded_array = np.load('./result/laiwen.npy')

    filtered_data = loaded_array[loaded_array[:, 1] != "GradientDescentRegression"]

    print(filtered_data)
            

    # print(loaded_array)
    np.save('./result/laiwen.npy', filtered_data)
    result_to_csv()


def modify_status(seq_type, reg_type, batch_size_option, epoch_option, status):

    loaded_array = np.load('./result/progress_status.npy')

    for item in loaded_array:
        if seq_type == item[0] and reg_type == item[1] and str(batch_size_option) == item[2] and str(epoch_option) == item[3]:

            item[4] = status
            break

    np.save('./result/progress_status.npy', loaded_array)
    result_to_csv()


def check_status(seq_type,reg_type, batch_size, epoch):

    loaded_array = np.load('./result/progress_status.npy')

    for item in loaded_array:
        if seq_type == item[0] and reg_type == item[1] and str(batch_size) == item[2] and str(epoch) == item[3]:
            return float(item[4])
        

def merge_npy():
    loaded1 = np.load('./result/progress_status.npy')
    loaded2 = np.load('./result/laiwen.npy')

    for i in range(loaded1.shape[0]):
        if loaded1[i, 4] != loaded2[i, 4] and loaded1[i, 4] == '0':
            loaded1[i, 4] = loaded2[i, 4]
            # print(f"第 {i+1} 列的第 5 個元素不同。")


    # np.save('./result/progress_status.npy', loaded1)
    # result_to_csv()            


def result_to_csv():
    # Load the .npy file
    data = np.load('./result/progress_status.npy')  # Replace with your .npy file path

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Save as .csv file
    df.to_csv('./result/progress_status.csv', index=False)  # Replace with desired .csv file name

def test():  
    # 載入 .npy 檔案
    data = np.load('./result/progress_status.npy')

    # 刪除最後一個元素
    data = data[:-1]
    print(data)
    # 儲存回 .npy 檔案
    np.save('./result/progress_status.npy', data)
    result_to_csv()   

# def test():

#     loaded_array = np.load('./result/progress_status.npy')
#     # 使用 np.char.find 檢查每行是否包含 "GRU"
#     contains_gru = np.char.find(loaded_array, "L") != -1  # 如果 "GRU" 存在，find 返回索引，否則返回 -1

#     # 根據條件刪除包含 "GRU" 的行
#     arr_filtered = loaded_array[~np.any(contains_gru, axis=1)]  # 刪除包含 "GRU" 的行

#     print("篩選後的陣列：")
#     print(arr_filtered)
#     np.save('./result/progress_status.npy', arr_filtered)
#     result_to_csv()
