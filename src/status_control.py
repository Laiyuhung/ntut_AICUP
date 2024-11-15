import itertools
import numpy as np
import pandas as pd
from itertools import product


def build_status(seq_model_type, reg_type, batch_size_option, epoch_option):

    loaded_array = np.load('./result/progress_status.npy')
    
    cartesian = np.array(list(product([seq_model_type],reg_type, batch_size_option, epoch_option)))
    status = 0

    # print(cartesian)
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

    loaded_array = np.load('./result/progress_status.npy')

    for item in loaded_array:
    #     # 想要手動更新時在這邊動作
    #     # item[0]:Model,
    #     # item[1]:Batch_size
    #     # item[2]:Epoch
    #     # item[3]:Status
    #     if item[0] == "r" and item[1] == "128" and item[2] == "50":
    #         item[3] = 3373929.26
    #     if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "100":
    #         item[3] = 6480500.77
    #     if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "150":
    #         item[3] = 4851553.75
    #     if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "200":
    #         item[3] = 3926538.65
    #     if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "250":
    #         item[3] = 3652284.36
    #     if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "300":
    #         item[3] = 5361942.5
            
        if item[0] == "LSTM":
            item[0] = "Transformer"

    print(loaded_array)
    # np.save('./result/progress_status.npy', loaded_array)
    # result_to_csv()


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
    loaded2 = np.load('./result/progress_status_outer.npy')

    for item1 in loaded1:
            for item2 in loaded2:
               if item1[4] == '0' and item2[4] != '0': 
                   item1[4] == item2[4]

    # np.save('./result/progress_status.npy', loaded1)
    # result_to_csv()            


def result_to_csv():
    # Load the .npy file
    data = np.load('./result/progress_status.npy')  # Replace with your .npy file path

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Save as .csv file
    df.to_csv('./result/progress_status.csv', index=False)  # Replace with desired .csv file name


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
