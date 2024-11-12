import itertools
import numpy as np
from itertools import product


def build_status(reg_type, batch_size_option, epoch_option):

    loaded_array = np.load('./result/progress_status.npy')
    
    cartesian = np.array(list(product([reg_type], batch_size_option, epoch_option)))
    status = 0
    
    # print(cartesian)

    for item in cartesian:
        cartesian_with_status = np.array([item[0], item[1], item[2], status])
        cartesian_with_status = cartesian_with_status.reshape(1, -1)
        print(cartesian_with_status)
        loaded_array = np.append(loaded_array, cartesian_with_status, axis=0)
    

    print(loaded_array)
    # print(cartesian_with_status)

    # loaded_array = np.append(loaded_array, cartesian_with_status, axis=0)
    # print(loaded_array)
    
    # loaded_array = np.delete(loaded_array, np.arange(-6, 0), axis=0)

    # print(loaded_array)


    np.save('./result/progress_status.npy', loaded_array)


def status_print():

    loaded_array = np.load('./result/progress_status.npy')
    print(loaded_array)

    min_error = 10000000
    
    for item in loaded_array:
        if float(item[3]) < min_error and float(item[3])!=0:
            min_combination = item
            min_error = float(item[3])

    print("Best: ",min_combination)

def manuel_modify_status():

    loaded_array = np.load('./result/progress_status.npy')

    for item in loaded_array:
        # 想要手動更新時在這邊動作
        # item[0]:Model,
        # item[1]:Batch_size
        # item[2]:Epoch
        # item[3]:Status
        if item[0] == "r" and item[1] == "128" and item[2] == "50":
            item[3] = 3373929.26
        if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "100":
            item[3] = 6480500.77
        if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "150":
            item[3] = 4851553.75
        if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "200":
            item[3] = 3926538.65
        if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "250":
            item[3] = 3652284.36
        if item[0] == "KnnRegression" and item[1] == "128" and item[2] == "300":
            item[3] = 5361942.5
            
    
    np.save('./result/progress_status.npy', loaded_array)


def modify_status(reg_type, batch_size_option, epoch_option, status):

    loaded_array = np.load('./result/progress_status.npy')

    for item in loaded_array:
        if reg_type == item[0] and str(batch_size_option) == item[1] and str(epoch_option) == item[2]:

            item[3] = status
            break

    np.save('./result/progress_status.npy', loaded_array)


def check_status(reg_type, batch_size, epoch):

    loaded_array = np.load('./result/progress_status.npy')

    for item in loaded_array:
        if reg_type == item[0] and str(batch_size) == item[1] and str(epoch) == item[2]:
            return float(item[3])
