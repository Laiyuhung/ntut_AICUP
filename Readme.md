### 環境

作業系統 : WSL2 Ubuntu 20.04.06 LTS

Python 3.12

### 安裝套件

```bash
    ./install.sh
```

### 執行程式

```bash
    ./run.sh
```

### 更改訓練batch size

在 ./src/main.py : 34 新增需要的batch size
```python=
    batch_size_option = [256, 128, 64]
```
### 更改訓練epoch

在 ./src/main.py : 35 新增需要的epoch
```python=
    epoch_option = [50, 100, 150, 200, 250, 300]
```

### 更換回歸模型
1. 新增.py檔撰寫並寫import到 ./src/main.py<br/>
在 ./src/main.py : 33 新增需要的模型
```python=
    reg_type = ["ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear"]
```
2. 在 ./src/main.py : 33 新增需要的模型
```python=
    reg_type = ["ExtraTreesRegressor", "KnnRegression", "VotingRegressor", "Linear"]
```
3. 在 ./src/main.py : 53 行開始處新增需要的模型
4. 手動新增需要的status記錄列
```
需將新增之回歸模型與reg_type、epoch_option做笛卡兒積(不要把原本的刪掉)，並且把status設為"0"
```


### 結果呈現
- output.csv 會存在result資料夾中
- output.csv 會存在result資料夾中丟給天氣小幫手前需要手動把 upload.csv 和 output.csv 的第一行（中文）刪除
- 執行./src/status_control.py 中的 status_print() 可得知目前預測進度
- progress_status.npy 用於儲存狀態之numpy array
