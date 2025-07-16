"""
剔除预测分数部分，RF部分，用于测试比较
Bayesian Optimization Script for Laser Powder Bed Fusion (LPBF) Parameter Optimization.

This script performs batch Bayesian optimization using both mechanical and surface quality
objectives. It integrates Gaussian Process (GP) surrogate modeling, a Random Forest
classifier for feasibility filtering, and qLogEHVI acquisition to select promising new
parameter sets.

Main steps:
1. Load initial experimental data.
2. Train surrogate models for mechanical and surface performance.
3. Predict missing (label_visibility, surface_uniformity) data with surrogate models.
4. Combine objectives via scalarization.
5. Perform batch Bayesian Optimization with feasibility filtering.

Author: Maoyurun Mao
Date: 07/16/2025
"""
import pandas as pd
import torch
from botorch.models import ModelListGP
from torch import Tensor

from models import SingleTaskGP_model
from optimization import qLogEHVI


def objective(f: torch.Tensor, weight: list) -> torch.Tensor:
    """
    Scalarize multi-objective outputs using a weighted sum.

    This function takes an [N, M] tensor of objective values and a list of weights
    for each objective, then computes a scalarized [N] objective value per sample.

    Args:
        f (torch.Tensor): Tensor of shape [N, M], where N is the number of samples,
                          and M is the number of objectives.
        weight (list): List or tensor of length M, specifying weights for each objective.
                       The weights do not need to sum to 1.

    Returns:
        torch.Tensor: Tensor of shape [N], representing the scalarized objective value
                      for each sample.
    """
    weight_tensor = torch.tensor(weight, dtype=f.dtype, device=f.device)
    value = (f * weight_tensor).sum(dim=-1)

    return value


def normalize_tensor(y: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor to [0, 1] range using min-max normalization.

    Args:
        y (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    return (y - y.min()) / (y.max() - y.min() + 1e-8)


def run_bo(
    model: ModelListGP,
    bounds: torch.Tensor,
    train_y: torch.Tensor,
    ref_point: list,
    batch_size: int,
    mini_batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Run batch Bayesian Optimization using qLogEHVI.

    Args:
        model (ModelListGP): Trained multi-objective GP model.
        bounds (torch.Tensor): Optimization variable bounds [2, d].
        train_y (torch.Tensor): Training objectives, shape [N, 2].
        ref_point (list): Reference point in objective space, e.g., [0.5, 0.5].
        batch_size (int): Target number of new samples to generate.
        mini_batch_size (int): BO internal batch size per iteration.
        device (torch.device): Target device (CPU/GPU).

    Returns:
        torch.Tensor: New candidate points, shape [batch_size, d].
    """
    X_next_tensor = torch.empty((0, bounds.shape[1]), dtype=torch.double).to(device)
    iteration = 0

    while X_next_tensor.shape[0] < batch_size:
        X_candidates, acq_val = qLogEHVI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=mini_batch_size
        )
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")
    return X_next_tensor


def read_data(filename,x_target: list,y_target: list, device: torch.device) -> tuple[Tensor, Tensor]:
    df = pd.read_csv(filename)
    x = torch.tensor(df[x_target].values, dtype=torch.double).to(device)
    y = torch.tensor(df[y_target].values, dtype=torch.double).to(device)
    return x, y


def main():
    # -------------------- 0. Initialization  -------------------- #
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
    # ref_point = torch.tensor([800, 12, 7, -0.15], dtype=torch.double).to(device)  # reference point for optimization


    # -------------------- 1. Initial Samples  -------------------- #
    # Read Samples data from old tasks
    X_target = ["power", "hatch_distance", "outline_power"]
    Y_target = [
        "label_visibility",
        "surface_uniformity",
        "Young's_modulus",
        "tensile_strength",
        "Elongation",
        "Edge_measurement"
    ]
    [X, Y] = read_data("data1.csv", X_target, Y_target, device)

    # Unified optimization direction
    # Initial data -> (N,1)
    y_label_visibility = Y[:, 0].unsqueeze(-1)
    y_surface_uniformity = Y[:, 1].unsqueeze(-1)
    y_E = Y[:, 2].unsqueeze(-1)
    y_strength = Y[:, 3].unsqueeze(-1)
    y_elongation = Y[:, 4].unsqueeze(-1)
    y_edge_error = Y[:, 5].unsqueeze(-1)

    # -------------------- 2. Random Forest Classifier  -------------------- #
    # X_RF = X[:, 0:2].cpu().numpy()
    # clf = rf_classifier.build_rf_classifier(X_RF, Fused_Label)


    # -------------------- 3. Surrogate Model  -------------------- #
    # 3.1 Mechanic GP models
    gp_E = SingleTaskGP_model.build_single_model(X, y_E)
    gp_strength = SingleTaskGP_model.build_single_model(X, y_strength)
    gp_elongation = SingleTaskGP_model.build_single_model(X, y_elongation)
    gp_edge_error = SingleTaskGP_model.build_single_model(X, y_edge_error)

    # 3.1.1 Scalarization
    # Normalization
    y_E_n = normalize_tensor(y_E)
    y_strength_n = normalize_tensor(y_strength)
    y_elongation_n = normalize_tensor(y_elongation)

    # Evaluating
    f_mecha = objective(
        torch.cat([y_E_n, y_strength_n, y_elongation_n], dim=1),
        weight=[0.34, 0.33, 0.33]
    ).unsqueeze(-1)
    gp_f_mecha = SingleTaskGP_model.build_single_model(X, f_mecha)

    # 3.2 Surface GP models
    # 3.2.1 Cleaning Data

    # 3.2.2 Scalarization
    # Normalization
    y_label_visibility_n = normalize_tensor(y_label_visibility)
    y_surface_uniformity_n = normalize_tensor(y_surface_uniformity)

    # Evaluating
    f_surface = objective(
        torch.cat([y_label_visibility_n, y_surface_uniformity_n], dim=1),
        weight=[0.5, 0.5]
    ).unsqueeze(-1)
    gp_f_surface = SingleTaskGP_model.build_single_model(X, f_surface)


    # -------------------- 4. Bayesian Optimization  -------------------- #
    model = ModelListGP(gp_f_mecha, gp_f_surface)
    Y_bo = torch.cat((f_mecha, f_surface), dim=1)
    X_next_tensor = run_bo(
        model=model,
        bounds=bounds,
        train_y=Y_bo,
        ref_point=[0.5, 0.5],
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        device=device
    )
    print(X_next_tensor)
pass

if __name__ == "__main__":
    main()
