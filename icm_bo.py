import pandas as pd
import torch
from botorch.models import SingleTaskGP, ModelListGP

from model import icm_model
from optimization import qLogEHVI
from model import black_box
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

# ---------- 1. Initial Samples  ---------- #
# Initial Samples from source task
df = pd.read_csv("data/source_task_data.csv")
X_source = torch.tensor(df[["P", "v", "t", "h"]].values, dtype=torch.double)
Y_source = torch.tensor(df[["Density", "Neg_Roughness", "Neg_Time"]].values, dtype=torch.double)
# Initial Samples from target task
df = pd.read_csv("data/target_task_data.csv")
X_target = torch.tensor(df[["P", "v", "t", "h"]].values, dtype=torch.double)
Y_target = torch.tensor(df[["Density", "Neg_Roughness", "Neg_Time"]].values, dtype=torch.double)

# ---------- 2. Bayesian Optimization Main Loop ---------- #
T = 10  # BO iteration times
batch_size = 1
hv_history = []
slack = [0.02, 0.1, 0.3]

ref_point = qLogEHVI.get_ref_point(Y_target, slack)
print("ref_point =", ref_point)
hv = Hypervolume(ref_point=ref_point)

for iteration in range(T):
    torch.cuda.empty_cache()
    print(f"\n========= Iteration {iteration + 1} =========")

    # 2.1 Build surrogate model
    model = icm_model.build_model(X_source, Y_source, X_target, Y_target)
    i: int = 0
    target_models = []
    for m in model.models:
        model_target = SingleTaskGP(X_target, Y_target[:, i:i + 1],
                                     input_transform=m.input_transform,
                                     outcome_transform=m.outcome_transform)
        model_target.load_state_dict(m.state_dict(), strict=False)
        target_models.append(model_target)
        i = i + 1
    model_for_bo = ModelListGP(*target_models)

    # 2.3 Optimize acquisition function and get next batch
    X_next, acq_val = qLogEHVI.optimize_acq_fun(
        model=model_for_bo,
        train_y=Y_target,
        bounds=bounds,
        batch_size=batch_size,
        ref_point=ref_point
    )

    # 2.4 Evaluate new points with a black-box function
    Y_next = black_box.func_2(X_next)

    # 2.5 Update datasets
    X_target = torch.cat([X_target, X_next], dim=0)
    Y_target = torch.cat([Y_target, Y_next], dim=0)
    # print current batch
    for i in range(batch_size):
        print(f"Candidate {i + 1}: X = {X_next[i].tolist()}, Y = {Y_next[i].tolist()}")

    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=Y_source
    )
    hv_value = hv.compute(partitioning.pareto_Y)
    print(hv_value)
    hv_history.append(hv_value)

df = pd.DataFrame(hv_history, columns=["HV"])
df.to_csv("result/warm_start_HV.csv", index=False)  # 可改路径
print("✅ Save in：result/warm_start_HV.csv")

plt.plot(hv_history, marker='o')
plt.title("Hyper volume Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Hyper volume")

plt.grid(True)
plt.tight_layout()
plt.show()
