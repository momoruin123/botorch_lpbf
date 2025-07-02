import pandas as pd
import torch
from models import SingleTaskGP_model
import matplotlib
from pathlib import Path

from optimization import qLogEHVI


matplotlib.use("TkAgg")
torch.manual_seed(42)

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [25, 0.1, 25],  # Upper bounds
    [300, 0.6, 300]  # Lower bounds
], dtype=torch.double)

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
ref_point = qLogEHVI.get_ref_point(Y, slack=0)
batch_size = 20
print(ref_point)

model = SingleTaskGP_model.build_model(X, Y)

X_next, acq_val = qLogEHVI.optimize_acq_fun(
    model=model,
    train_y=Y,
    bounds=bounds,
    batch_size=batch_size,
    ref_point=ref_point
)

print(X_next)