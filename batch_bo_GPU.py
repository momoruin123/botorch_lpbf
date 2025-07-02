import pandas as pd
import torch
from models import SingleTaskGP_model
from optimization import qLogEHVI
from models import black_box
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [50, 200, 0.09, 0.1],  # Upper bounds
    [150, 1000, 0.13, 0.4]  # Lower bounds
], dtype=torch.double)
bounds = bounds.to(device)

# ---------- 1. Initial Samples  ---------- #
# Initial Samples from target task
df = pd.read_csv("D:/botorch_lpbf/data/target_task_data.csv")
X_train_ = torch.tensor(df[["P", "v", "t", "h"]].values, dtype=torch.double)
Y_train_ = torch.tensor(df[["Density", "Neg_Roughness", "Neg_Time"]].values, dtype=torch.double)
# Merge set
X_train = X_train_.to(device)
Y_train = Y_train_.to(device)

# ---------- 2. Bayesian Optimization Main Loop ---------- #
T = 10  # BO epoch times
batch_size = 1
hv_history = []
slack=[0.01, 0.3, 0.5]

ref_point = qLogEHVI.get_ref_point(Y_train, slack)
ref_point = torch.tensor([  0.8253,  -7.0257, -19.1959], device='cuda:0', dtype=torch.float64)
ref_point = ref_point.to(device)
hv = Hypervolume(ref_point=ref_point)

for iteration in range(T):
    torch.cuda.empty_cache()
    print(f"\n========= Iteration {iteration + 1} =========")

    # 2.1 Build surrogate model
    model = gp_model.build_model(X_train, Y_train)
    model = model.to(device)

    # 2.3 Optimize acquisition function and get next batch
    X_next, acq_val = qLogEHVI.optimize_acq_fun(
        model=model,
        train_y=Y_train,
        bounds=bounds,
        batch_size=batch_size,
        ref_point=ref_point
    )
    X_next = X_next.to(device)

    # 2.4 Evaluate new points with a black-box function
    Y_next = black_box.func_2(X_next)

    # 2.5 Update datasets
    X_train = torch.cat([X_train, X_next], dim=0)
    Y_train = torch.cat([Y_train, Y_next], dim=0)
    # print current batch
    for i in range(batch_size):
        print(f"Candidate {i + 1}: X = {X_next[i].tolist()}, Y = {Y_next[i].tolist()}")

    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=Y_train
    )
    hv_value = hv.compute(partitioning.pareto_Y)
    print(hv_value)
    hv_history.append(hv_value)

df = pd.DataFrame(hv_history, columns=["HV"])
df.to_csv("result/cold_start_HV.csv", index=False)
print("✅ Save in：result/cold_start_HV.csv")

plt.plot(hv_history, marker='o')
plt.title("Hyper volume Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Hyper volume")
plt.grid(True)
plt.tight_layout()
plt.savefig("result/cold_start_HV.png")
plt.show()