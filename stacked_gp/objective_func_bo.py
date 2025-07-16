"""
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


import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from models import rf_classifier
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
    ref_point = torch.tensor([800, 12, 7, -0.15], dtype=torch.double).to(device)  # reference point for optimization


    # -------------------- 1. Initial Samples  -------------------- #
    # Read Samples data from old tasks
    df = pd.read_csv("data1.csv", encoding="utf-8-sig")
    X = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double).to(device)
    target = [
        "label_visibility",
        "surface_uniformity",
        "Youngs_modulus",
        "tensile_strength",
        "Elongation",
        "Edge_measurement"
    ]
    Y = torch.tensor(df[target].values, dtype=torch.double).to(device)
    Fused_Label = torch.tensor(df[["fused"]].values, dtype=torch.double).squeeze().cpu().numpy()
    id = torch.tensor(df[["sample_id"]].values, dtype=torch.int).squeeze().cpu().numpy()

    # Initial data -> (N,1)
    y_label_visibility = Y[:, 0].unsqueeze(-1)
    y_surface_uniformity = Y[:, 1].unsqueeze(-1)
    y_E = Y[:, 2].unsqueeze(-1)
    y_strength = Y[:, 3].unsqueeze(-1)
    y_elongation = Y[:, 4].unsqueeze(-1)
    y_edge_error = -Y[:, 5].unsqueeze(-1)

    # -------------------- 2. Random Forest Classifier  -------------------- #
    X_RF = X[:, 0:2].cpu().numpy()
    clf = rf_classifier.build_rf_classifier(X_RF, Fused_Label)


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
    # choose id > 20
    mask_id = id > 20
    y_label_visibility_20 = y_label_visibility[~mask_id]  # save y_i(i<20)
    y_surface_uniformity_20 = y_surface_uniformity[~mask_id]
    y_label_visibility = y_label_visibility[mask_id]
    y_surface_uniformity = y_surface_uniformity[mask_id]
    x_surface_evaluation = Y[:, 2:5][mask_id]
    # filter NAN
    mask_nan = ~torch.isnan(y_label_visibility).squeeze()
    y_label_visibility = y_label_visibility[mask_nan]
    y_surface_uniformity = y_surface_uniformity[mask_nan]
    x_surface_evaluation_nan = x_surface_evaluation[~mask_nan]  # save x of NAN
    x_surface_evaluation = x_surface_evaluation[mask_nan]  # filter x of NAN
    # build GP(x_surface_evaluation -> y)
    gp_label_visibility = SingleTaskGP_model.build_single_model(x_surface_evaluation, y_label_visibility)
    gp_surface_uniformity = SingleTaskGP_model.build_single_model(x_surface_evaluation, y_surface_uniformity)
    # predict batch 4 by mechanic measurements, y = GP(x_surface_evaluation_nan)
    y_label_visibility_pred = SingleTaskGP_model.predict(gp_label_visibility, x_surface_evaluation_nan)
    y_surface_uniformity_pred = SingleTaskGP_model.predict(gp_surface_uniformity, x_surface_evaluation_nan)
    # merge data
    y_label_visibility = torch.cat((y_label_visibility_20, y_label_visibility, y_label_visibility_pred), dim=0)
    y_surface_uniformity = torch.cat((y_surface_uniformity_20, y_surface_uniformity, y_surface_uniformity_pred), dim=0)

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
    ref_point = [0.5, 0.5]
    X_next_np = np.empty((0, X.shape[1]))
    while len(X_next_np) < batch_size:
        X_candidates, acq_val = qLogEHVI.optimize_acq_fun(
            model=model,
            train_y=Y_bo,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=mini_batch_size
        )
        X_candidates_np = X_candidates.detach().cpu().numpy()
        preds = clf.predict(X_candidates_np[:, 0:2])  # shape = [batch_size]
        fused_mask = preds == 1
        fused_points = X_candidates_np[fused_mask]
        print(fused_points)
        X_next_np = np.vstack([X_next_np, fused_points])
    X_next_tensor = torch.tensor(X_next_np, dtype=torch.double)
    print(X_next_tensor)
pass

if __name__ == "__main__":
    main()
