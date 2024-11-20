import csv

def compare_csv(file1, file2):
    with open(file1, 'r', newline='', encoding='utf-8') as f1, open(file2, 'r', newline='', encoding='utf-8') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        
        # 讀取 CSV 內容
        rows1 = list(reader1)
        rows2 = list(reader2)
        sum = 0
        # 逐行比對第二欄
        for i, (row1, row2) in enumerate(zip(rows1, rows2), start=1):
            if len(row1) > 1 and len(row2) > 1:  # 確保有第二欄
                if row1[1] != row2[1]:
                    print(f"Line {i} differs: {file1} -> '{row2[1]}' - {file2} -> '{row1[1]}' = {float(row2[1]) - float(row1[1])}")
                    sum = sum + abs(float(row2[1]) - float(row1[1]))

        print("total difference:", sum)

# 使用範例
file1 = './result/competition_output.csv'
file2 = './result/ria.csv'
compare_csv(file1, file2)
