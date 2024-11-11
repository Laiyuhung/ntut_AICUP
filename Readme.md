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

### 更改訓練圈數

./src/main.py : 20
```python=
    train( X_train , y_train , NowDateTime , batch_size = 128 , epochs = 次數 )
```
#### 更換模型
可在 ./src/main.py 最上面 from xxxRegression import *
中選擇回歸模型

- 1. Regression : 線性回歸
- 2. ExtraTreesRegressor : 極限隨機樹回歸

Output.csv 會在同層
丟給天氣小幫手前需要手動把 upload.csv 和 output.csv 的第一行（中文）刪除
