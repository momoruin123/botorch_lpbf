import torch
import pandas as pd
from models import black_box

torch.manual_seed(42)  # fix random seed

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [50, 200, 0.09, 0.1],  # Upper bounds
    [150, 1000, 0.13, 0.4]  # Lower bounds
], dtype=torch.double)

N_init = 12
X_train = torch.rand(N_init, 4) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.func1(X_train)

data = torch.cat([X_train, Y_train], dim=1)
df = pd.DataFrame(data.numpy(), columns=["P", "v", "t", "h", "Density", "Neg_Roughness", "Neg_Time"])
df.to_csv("data/source_task_data.csv", index=False)  # 可改路径
print("✅ Save in：data/source_task_data.csv")

N_init = 4
X_train = torch.rand(N_init, 4) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.func2(X_train)

data = torch.cat([X_train, Y_train], dim=1)
df = pd.DataFrame(data.numpy(), columns=["P", "v", "t", "h", "Density", "Neg_Roughness", "Neg_Time"])
df.to_csv("data/target_task_data.csv", index=False)  # 可改路径
print("✅ Save in：data/target_task_data.csv")