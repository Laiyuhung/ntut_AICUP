import pandas as pd
import numpy as np

# 使用 pandas 讀取 CSV 文件
df1 = pd.read_csv('./data/ExampleTrainData(AVG)/TotalCloudAmount/Hualian_TotalCloudAmount_combined.csv')

# 將 DataFrame 轉換為 numpy array
data = df1.to_numpy()
# print(data)

np.save('./data/ExampleTrainData(AVG)/TotalCloudAmount/Hualian_TotalCloudAmount.npy', data)




data = np.load('./data/ExampleTrainData(AVG)/TotalCloudAmount/Hualian_TotalCloudAmount.npy')  # 替換 'your_file.npy' 為你的 .npy 文件名稱

# for row in data:
#     print(row)
#     print(row[0])


# 使用 pandas 讀取 CSV 文件
df = pd.read_csv('./data/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_17_modified2.csv')

# 判斷並修改第五列的值
# 假設第五列的標題是 "column_5"
# 如果 CSV 沒有標題，你可以使用索引 `df.iloc[:, 4]` 來訪問第五列
for index, value in df.iloc[:, 7].items():  # 使用 iloc 訪問第五列（索引從 0 開始，所以第五列是 4）
    # print(value)
    print(index)
    for row in data:
        if value == row[0]: # date確認
            print(df.iloc[index, 8])
            df.iloc[index, 12] = row[df.iloc[index, 8]] # hour確認 (12: cloud)
            # print(df.iloc[index, 12])
            break


# 逐行列印修改後的內容
for index, row in df.iterrows():
    print(row.to_list())

# 將修改後的數據保存回 CSV
df.to_csv('./data/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_17_modified3.csv', index=False)
