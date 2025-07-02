import torch
import pandas as pd

x_new1 = torch.tensor([
    [6.3265e+01, 1.0704e-01, 2.5000e+01],
    [2.5000e+01, 1.0000e-01, 2.5000e+01],
    [4.2577e+01, 1.6563e-01, 1.1255e+02],
    [1.2055e+02, 2.0673e-01, 2.8106e+01],
    [2.1368e+02, 1.0000e-01, 7.8716e+01],
    [5.1581e+01, 1.0000e-01, 2.5000e+01],
    [3.5985e+01, 1.5497e-01, 4.4081e+01],
    [2.0204e+02, 1.0000e-01, 5.7662e+01],
    [5.3703e+01, 1.1819e-01, 2.5000e+01],
    [3.9497e+01, 1.4439e-01, 1.3874e+02]
], dtype=torch.float64)

x_new2 = torch.tensor([
    [5.6281e+01, 1.0803e-01, 2.5000e+01],
    [2.5000e+01, 1.0000e-01, 2.5000e+01],
    [2.5572e+01, 1.5718e-01, 1.1780e+02],
    [5.7270e+01, 1.1937e-01, 2.5000e+01],
    [3.0000e+02, 1.0000e-01, 2.5000e+01],
    [6.4586e+01, 1.0000e-01, 2.5000e+01],
    [5.1007e+01, 1.0000e-01, 3.0987e+01],
    [7.3615e+01, 1.1414e-01, 2.5000e+01],
    [3.3002e+01, 1.0000e-01, 4.4826e+01],
    [5.7018e+01, 1.0988e-01, 2.5000e+01]
], dtype=torch.float64)

# 转换并保留两位小数
def format_tensor(tensor):
    return [[round(float(x), 2) for x in row] for row in tensor.tolist()]

x_new1_list = format_tensor(x_new1)
x_new2_list = format_tensor(x_new2)
X_new = x_new1_list+ x_new2_list
print("x_new:", X_new)

df = pd.DataFrame(X_new, columns=["power", "hatch_distance", "outline_power"])
df.to_csv("Samples.csv", index=False)  # 可改路径
print("✅ Save in：Samples.csv")
print("保存成功：Samples.xlsx")