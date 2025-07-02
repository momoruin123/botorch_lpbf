import torch
from models import SingleTaskGP_model
from optimization import qLogEHVI

# 假设 train_X 形状为 (N, 4) -> 4个工艺参数
train_x = torch.tensor([
    [100, 800, 0.1, 0.08],  # [power, speed, layer thickness, hatch_spacing]
    [120, 1000, 0.1, 0.10],
    [80, 900, 0.12, 0.07],
    [110, 1100, 0.1, 0.09],
    [95, 850, 0.11, 0.08],
    [105, 950, 0.1, 0.07],
    [90, 1000, 0.13, 0.1],
    [115, 970, 0.09, 0.08],
], dtype=torch.double)

# train_Y shape = (N, 3) -> 三个目标
train_y = torch.tensor([
    [0.98, 8.5, 12.0],  # [Density, Roughness, Time]
    [0.96, 9.0, 10.5],
    [0.92, 10.5, 11.5],
    [0.97, 8.0, 10.0],
    [0.95, 9.3, 12.2],
    [0.96, 9.1, 11.0],
    [0.93, 10.0, 12.5],
    [0.97, 8.7, 10.3],
], dtype=torch.double)

model = gp_model.build_model(train_x, train_y)

ref_point = [0.9, 11.0, 13.0]  # [density (min), roughness (max), time (max)]

bounds = [
    [80, 800, 0.09, 0.07],
    [120, 1100, 0.13, 0.10]
]

sample, a = qLogEHVI.optimize_acq_fun(model, train_y, bounds, batch_size=3, ref_point=ref_point)
print(sample)
print(a)
