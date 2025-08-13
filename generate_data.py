import torch
import pandas as pd
from models import black_box

torch.manual_seed(42)  # fix random seed

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [0, 0, 0, 0, 0],  # lower bounds
    [1, 1, 1, 1, 1]  # upper bounds
], dtype=torch.double)

N_init = 100
X_train = torch.rand(N_init, 5) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.transfer_model_1(X_train, 5)

data = torch.cat([X_train, Y_train], dim=1)
df = pd.DataFrame(data.numpy())
df.to_csv("data/source_task_data.csv", index=False)  # 可改路径
print("Save in：data/source_task_data.csv")

N_init = 20
X_train = torch.rand(N_init, 5) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.transfer_model_2(X_train, 5)

data = torch.cat([X_train, Y_train], dim=1)
df = pd.DataFrame(data.numpy(), columns=None)
df.to_csv("data/target_task_data.csv", index=False)  # 可改路径
print("Save in：data/target_task_data.csv")