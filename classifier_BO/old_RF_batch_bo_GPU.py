import pandas as pd
import torch
from botorch.optim import optimize_acqf
from models import SingleTaskGP_model
import matplotlib
from pathlib import Path

from optimization.qLogEHVI import build_acq_fun


def fused_constraint(x: torch.Tensor) -> torch.Tensor:
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    power = x[..., 0]
    hatch = x[..., 1]

    # 条件1: power >= 200
    cond1 = power - 200  # 满足约束时 >= 0

    # 条件2: (hatch <= 0.25) & (power < 200)
    cond2a = 0.25 - hatch  # hatch <= 0.25 => cond2a >= 0
    cond2b = 200 - power   # power < 200 => cond2b > 0

    # 取两个中的最小值，表示两者都满足才算满足
    cond2 = torch.minimum(cond2a, cond2b)

    # 两种情况，只要满足一种就行：最大值即可
    constraint_val = torch.maximum(cond1, cond2)

    return constraint_val


def get_initial_conditions(num_restarts, bounds, batch_size):
    def _is_fused_region(x):
        power = x[:, 0]
        hatch = x[:, 1]
        mask = (power >= 200) | ((hatch <= 0.25) & (power < 200))
        return mask

    d = bounds.shape[1]
    x_init = []
    device = bounds.device

    while len(x_init) < num_restarts:
        candidate = torch.rand(batch_size, d, dtype=torch.double, requires_grad=True).to(device)
        candidate = bounds[0] + (bounds[1] - bounds[0]) * candidate

        mask = _is_fused_region(candidate)
        valid_points = candidate[mask]

        for vp in valid_points:
            if len(x_init) >= num_restarts:
                break
            x_init.append(vp.unsqueeze(0))

    x_init = torch.cat(x_init, dim=0).unsqueeze(1)
    return x_init


matplotlib.use("TkAgg")
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [25, 0.1, 25],  # Upper bounds
    [300, 0.6, 300]  # Lower bounds
], dtype=torch.double)
bounds = bounds.to(device)

# ---------- 1. Initial Samples  ---------- #
# Initial Samples from target task
current_dir = Path.cwd()
csv_path = current_dir.parent / "data" / "classifier_BO.csv"
df = pd.read_csv(csv_path)
X = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double)
Y = torch.tensor(df[["fused", "edge_clarity", "label_visibility", "surface_uniformity"]].values, dtype=torch.double)
Y = Y[:, 1:]  # only need the last three columns
print(X.shape, Y.shape)

# ---------- 2. Bayesian Optimization  ---------- #
ref_point = torch.tensor([5, 5, 5], dtype=torch.double)
batch_size = 20
print(ref_point)

model = SingleTaskGP_model.build_model(X, Y)

# 2.3 Optimize acquisition function and get next batch
batch_initial_conditions = get_initial_conditions(50, bounds, batch_size)
print(batch_initial_conditions)
acq_func = build_acq_fun(model, ref_point, Y)

candidate, acq_value = optimize_acqf(
    acq_function=acq_func,
    bounds=bounds,
    q=batch_size,
    num_restarts=50,
    raw_samples=1024,
    return_best_only=True,
    batch_initial_conditions=batch_initial_conditions,
    nonlinear_inequality_constraints=[
        (fused_constraint, True)
    ]
)

print(candidate)
