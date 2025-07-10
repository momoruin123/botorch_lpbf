import pandas as pd
import torch
from models import rf_classifier
from models import SingleTaskGP_model
from optimization import qEI


def mechanic_objective(f: torch.Tensor, weight: list) -> torch.Tensor:
    weight_tensor = torch.tensor(weight, dtype=f.dtype, device=f.device)  # To tensor
    f_mechanic = (f * weight_tensor).sum(dim=-1)  # [N, 4] × [4] → [N,1]

    return f_mechanic


def normalize_tensor(y: torch.Tensor) -> torch.Tensor:
    return (y - y.min()) / (y.max() - y.min() + 1e-8)

def main():
    # ---------- 0. Initialization  ---------- #
    # matplotlib.use("TkAgg")  # Fix compatibility issues between matplotlib and botorch
    # torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 0.1 Set constance and Hyper parameters
    # Set parameters limit（Power, Hatch_distance）
    bounds = torch.tensor([
        [25, 0.1, 25],  # Lower bounds
        [300, 0.6, 300]  # Upper bounds
    ], dtype=torch.double).to(device)

    # 0.2 Set BO parameters
    batch_size = 20  # the finial batch size
    mini_batch_size = 10  # If computer is not performing well (smaller than batch_size)
    ref_point = torch.tensor([800, 12, 7, -0.15], dtype=torch.double).to(device)  # reference point for optimization

    # ---------- 1. Initial Samples  ---------- #
    # Initial Samples from old tasks
    df = pd.read_csv("data.csv")
    X = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double).to(device)
    target = ["Young's modulus", "tensile strength", "Elongation", "Edge measurement"]
    Y = torch.tensor(df[target].values, dtype=torch.double).to(device)
    Y[:, 3] = -Y[:, 3]
    Fused_Label = torch.tensor(df[["fused"]].values, dtype=torch.double).squeeze().cpu().numpy()
    Objective = torch.tensor(df[["objective"]].values, dtype=torch.double).squeeze().cpu().numpy()

    y_E = Y[:, 0].unsqueeze(-1)
    y_strength = Y[:, 1].unsqueeze(-1)
    y_elongation = Y[:, 2].unsqueeze(-1)
    y_edge_error = Y[:, 3].unsqueeze(-1)
    # y_visible = Y[:, 4]

    # ---------- 2. Random Forest Classifier  ---------- #
    X_RF = X[:, 0:2].cpu().numpy()
    clf = rf_classifier.build_rf_classifier(X_RF, Fused_Label)

    # ---------- 3. Surrogate Model  ---------- #
    # 3.1 Mechanic GP models
    gp_E = SingleTaskGP_model.build_single_model(X, y_E)
    gp_strength = SingleTaskGP_model.build_single_model(X, y_strength)
    gp_elongation = SingleTaskGP_model.build_single_model(X, y_elongation)
    gp_edge_error = SingleTaskGP_model.build_single_model(X, y_edge_error)

    y_E_pred = SingleTaskGP_model.predict(gp_E, X)
    diff = y_E - y_E_pred
    print(diff)

    y_E_n = normalize_tensor(y_E)
    y_strength_n = normalize_tensor(y_strength)
    y_elongation_n = normalize_tensor(y_elongation)
    y_edge_error_n = normalize_tensor(y_edge_error)

    f_mecha = mechanic_objective(
        torch.cat([y_E_n, y_strength_n, y_elongation_n, y_edge_error_n], dim=1),
        weight=[0.3, 0.3, 0.2, 0.2]
    )
    gp_f_mecha = SingleTaskGP_model.build_single_model(X, f_mecha)

    # 2.3 Optimize acquisition function and get next batch
    X_next, acq_val = qEI.optimize_acq_fun(
        model=gp_f_mecha,
        train_y=f_mecha,
        bounds=bounds,
        batch_size=batch_size
    )
    X_next = X_next.to(device)
    print(X_next)
    pass


if __name__ == "__main__":
    main()
