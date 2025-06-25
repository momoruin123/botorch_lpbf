import torch
from models import gp_model
from optimization import qLogEHVI
from models import black_box
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
torch.manual_seed(42)

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [50, 200, 0.09, 0.1],  # Upper
    [150, 1000, 0.13, 0.4]  #
], dtype=torch.double)

# ---------- 1. Initial Samples ---------- #
N_init = 16
X_train = torch.rand(N_init, 4) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.func(X_train)
# ---------- 2. Bayesian Optimization Main Loop ---------- #
T = 5  # BO epoch times
batch_size = 5
hv_history = []
slack=[0.01, 0.5, 0.5]

ref_point = qLogEHVI.get_ref_point(Y_train, slack)
hv = Hypervolume(ref_point=torch.tensor(ref_point, dtype=torch.double))

for iteration in range(T):
    print(f"\n========= Iteration {iteration + 1} =========")

    # 2.1 Build surrogate model
    model = gp_model.build_model(X_train, Y_train)

    # 2.3 Optimize acquisition function and get next batch
    X_next, acq_val = qLogEHVI.optimize_acq_fun(
        model=model,
        train_y=Y_train,
        bounds=bounds,
        batch_size=batch_size,
        ref_point=ref_point
    )

    # 2.4 Evaluate new points with a black-box function
    Y_next = black_box.func(X_next)

    # 2.5 Update datasets
    X_train = torch.cat([X_train, X_next], dim=0)
    Y_train = torch.cat([Y_train, Y_next], dim=0)
    ref_point = qLogEHVI.get_ref_point(Y_train, slack)
    # print current batch
    for i in range(batch_size):
        print(f"Candidate {i + 1}: X = {X_next[i].tolist()}, Y = {Y_next[i].tolist()}")

    partitioning = NondominatedPartitioning(
        ref_point=torch.tensor(ref_point, dtype=torch.double),
        Y=Y_train
    )
    hv_value = hv.compute(partitioning.pareto_Y)
    hv_history.append(hv_value)

plt.plot(hv_history, marker='o')
plt.title("Hypervolume Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Hypervolume")
plt.grid(True)
plt.tight_layout()
plt.show()
