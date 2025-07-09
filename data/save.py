import torch
import pandas as pd

x_new1 = torch.tensor([[2.0845e+02, 4.0646e-01, 1.7300e+02],
        [2.1828e+02, 3.0523e-01, 1.1458e+02],
        [2.4480e+02, 4.5411e-01, 2.4329e+02],
        [2.4750e+02, 4.3673e-01, 2.0636e+02],
        [2.3355e+02, 3.8295e-01, 2.9204e+02],
        [1.1525e+02, 2.1938e-01, 1.6747e+02],
        [1.6242e+02, 5.8564e-01, 1.4549e+02],
        [2.7540e+02, 6.0000e-01, 1.5100e+02],
        [2.9036e+02, 5.3227e-01, 2.5409e+02],
        [2.3884e+02, 1.9183e-01, 1.4998e+02],
        [1.1491e+02, 1.0000e-01, 1.8620e+02],
        [2.2740e+02, 5.0309e-01, 2.0297e+02],
        [2.7830e+02, 6.0000e-01, 1.5498e+02],
        [1.6186e+02, 1.0000e-01, 7.8751e+01],
        [5.7582e+01, 6.0000e-01, 1.9877e+02],
        [3.1494e+01, 5.0030e-01, 2.2273e+02],
        [2.3214e+02, 3.2065e-01, 4.4896e+01],
        [4.4232e+01, 6.0000e-01, 8.0674e+01],
        [2.4592e+02, 5.4486e-01, 1.9531e+02],
        [2.6680e+02, 1.0000e-01, 1.8775e+02],
        [1.7002e+02, 1.9020e-01, 2.5188e+02],
        [2.8177e+02, 6.0000e-01, 1.4168e+02],
        [2.4604e+02, 6.0000e-01, 1.5059e+02]], dtype=torch.float64)

# 转换并保留两位小数
def format_tensor(tensor):
    return [[round(float(x), 2) for x in row] for row in tensor.tolist()]

x_new1_list = format_tensor(x_new1)
# x_new2_list = format_tensor(x_new2)
X_new = x_new1_list
print("x_new:", X_new)

df = pd.DataFrame(X_new, columns=["power", "hatch_distance", "outline_power"])
df.to_csv("Batch5_2.csv", index=False)  # 可改路径
print("✅ Save in：Samples.csv")
print("保存成功：Samples.xlsx")