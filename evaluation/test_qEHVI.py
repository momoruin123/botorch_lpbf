from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, qLogExpectedHypervolumeImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from models import gp_model
import torch

train_x = torch.tensor([
    [100, 800, 0.1, 0.08],
    [120, 1000, 0.1, 0.10],
    [80,  900, 0.12, 0.07],
    [110, 1100, 0.1, 0.09],
    [95,  850, 0.11, 0.08],
    [105, 950, 0.1, 0.07],
    [90,  1000, 0.13, 0.1],
    [115, 970, 0.09, 0.08],
], dtype=torch.double)

# train_Y shape = (N, 3) -> 三个目标
train_y = torch.tensor([
    [0.98, 8.5, 12.0],   # [Density, Roughness, Time]
    [0.96, 9.0, 10.5],
    [0.92, 10.5, 11.5],
    [0.97, 8.0, 10.0],
    [0.95, 9.3, 12.2],
    [0.96, 9.1, 11.0],
    [0.93, 10.0, 12.5],
    [0.97, 8.7, 10.3],
], dtype=torch.double)


model = gp_model.build_model(train_x, train_y)
# 用于采样的 MC 采样器
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=42)
# 你需要一个当前帕累托前沿上的参考点（略劣于最差目标）
ref_point = [0.9, 11.0, 13.0]  # [density (min), roughness (max), time (max)]

# identity objective 默认直接输出每一维目标
objective = IdentityMCMultiOutputObjective()

# 你必须先构造当前的 pareto 前沿（已知训练目标）
Y_pareto = train_y  # shape: (N, 3)

partitioning = NondominatedPartitioning(
    ref_point=torch.tensor([0.9, 11.0, 13.0], dtype=torch.double),
    Y=Y_pareto
)


acq_func = qLogExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    partitioning=partitioning,
    sampler=sampler,
    objective=objective
)

bounds = torch.stack([
    torch.tensor([80, 800, 0.09, 0.07], dtype=torch.double),
    torch.tensor([120, 1100, 0.13, 0.10], dtype=torch.double),
])

candidate, acq_value = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=3,  # batch size: 建议生成几个建议点
    num_restarts=10,
    raw_samples=100,
    return_best_only=True,
)

print("Next batch (P, v, t, h) :")
print(candidate)