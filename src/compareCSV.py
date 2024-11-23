import csv

def compare_csv(compare_name, best_name):

    file1 = f'./result/{compare_name}.csv'
    file2 = f'./result/{best_name}.csv'

    with open(file1, 'r', newline='', encoding='utf-8') as f1, open(file2, 'r', newline='', encoding='utf-8') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        
        # 讀取 CSV 內容
        rows1 = list(reader1)[1:]
        rows2 = list(reader2)[1:]
        sum = 0
        # 逐行比對第二欄
        for i, (row1, row2) in enumerate(zip(rows1, rows2), start=1):
            if len(row1) > 1 and len(row2) > 1:  # 確保有第二欄
                if row1[1] != row2[1]:
                    print(f"Line {i} differs: {file1} -> '{row2[1]}' - {file2} -> '{row1[1]}' = {float(row2[1]) - float(row1[1])}")
                    sum = sum + abs(float(row2[1]) - float(row1[1]))

        print("total difference:", sum)
        print("compare file: ",file1)
        print("best file: ",file2)

        with open('./result/ResultComparison.csv', 'a', newline='', encoding='utf-8') as out_csv:
            writer = csv.writer(out_csv)
            # writer.writerow(["compare_file", "best_file", "total_difference"])
            writer.writerow([file1, file2, sum])

# 使用範例

compare_name = "65_output_Transformer_Voting_256_150"
best_name = "best"
compare_csv(compare_name, best_name)
