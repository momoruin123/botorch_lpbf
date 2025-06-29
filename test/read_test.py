import pandas as pd
import torch


# 假设你的文件名是
df = pd.read_csv("D:/botorch_lpbf/data/target_task_data.csv", nrows=4)  # 路径可以是相对或绝对

# 提取输入参数和目标值
X = torch.tensor(df[["P", "v", "t", "h"]].values, dtype=torch.double)
Y = torch.tensor(df[["Density", "Neg_Roughness", "Neg_Time"]].values, dtype=torch.double)

print(X.shape)
print(Y.shape)
print(X)
print(Y)