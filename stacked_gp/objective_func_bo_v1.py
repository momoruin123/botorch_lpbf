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
from typing import Tuple

import pandas as pd
import torch
from botorch.utils.multi_objective import is_non_dominated
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor
from botorch.models import ModelListGP
from botorch.utils.sampling import draw_sobol_samples

from evaluation import bo_evaluation
from models import SingleTaskGP_model
from optimization import qLogEHVI
from models import black_box


def generate_initial_data(bounds: torch.Tensor, n_init: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    use Sobol 序列在给定 bounds 中生成初始样本，并用黑盒函数计算标签。

    Args:
        bounds (torch.Tensor): shape [2, d]，下限和上限
        n_init (int): 初始样本数量
        device (torch.device): 使用的设备

    Returns:
        Tuple of tensors: (X_init, Y_init)
    """
    sobol_X = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1).to(device)
    Y = black_box.mechanical_model(sobol_X)
    return sobol_X, Y


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
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")
    return X_next_tensor[:batch_size, :]


def read_data(filename, x_target: list, y_target: list, device: torch.device) -> tuple[Tensor, Tensor]:
    df = pd.read_csv(filename)
    x = torch.tensor(df[x_target].values, dtype=torch.double).to(device)
    y = torch.tensor(df[y_target].values, dtype=torch.double).to(device)
    return x, y


def main():
    # matplotlib.use("TkAgg")  # Fix compatibility issues between matplotlib and botorch
    # torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    hv_history = []  # log of Hyper volume
    gd_history = []  # # log of Hyper volume
    igd_history = []
    spacing_history = []
    cardinality_history = []
    # -------------------- 0. Initialization  -------------------- #
    # 0.1 Set constance and Hyper parameters
    # Set parameters limit（Power, Hatch_distance）
    bounds = torch.tensor([
        [25, 0.1, 25],  # Lower bounds
        [300, 0.6, 300]  # Upper bounds
    ], dtype=torch.double).to(device)

    # 0.2 Set BO parameters
    batch_size = 20  # the finial batch size
    mini_batch_size = 10  # If computer is not performing well (smaller than batch_size)

    # get true Pareto frontier
    X_ref, Y_ref = generate_initial_data(bounds=bounds, n_init=1000, device=device)  # [1000, M]
    mask_ref = is_non_dominated(Y_ref)
    true_pf = Y_ref[mask_ref]  # [P, M]

    # -------------------- 1. Initial Samples  -------------------- #
    n_init = batch_size  # 初始样本数
    X, Y = generate_initial_data(bounds=bounds, n_init=n_init, device=device)

    # Read Samples data from old tasks
    # X_target = ["power", "hatch_distance", "outline_power"]
    # Y_target = [
    #     "label_visibility",
    #     "surface_uniformity",
    #     "Young's_modulus",
    #     "tensile_strength",
    #     "Elongation",
    #     "Edge_measurement"
    # ]
    # [X, Y] = read_data("data1.csv", X_target, Y_target, device)

    # Unified optimization direction
    # Initial data -> (N,1)
    y_label_visibility = Y[:, 0].unsqueeze(-1)
    y_surface_uniformity = Y[:, 1].unsqueeze(-1)
    y_E = Y[:, 2].unsqueeze(-1)
    y_strength = Y[:, 3].unsqueeze(-1)
    y_elongation = Y[:, 4].unsqueeze(-1)
    # y_edge_error = -Y[:, 5].unsqueeze(-1)

    # -------------------- 2. Random Forest Classifier  -------------------- #
    # X_RF = X[:, 0:2].cpu().numpy()
    # clf = rf_classifier.build_rf_classifier(X_RF, Fused_Label)

    # -------------------- 3. Surrogate Model  -------------------- #
    n_iter = 5  # 迭代次数
    for i in range(n_iter):
        print(f"\n========= Iteration {i + 1}/{n_iter} =========")
        # Mechanics model
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

        # Surface model
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
        ref_point = qLogEHVI.get_ref_point(Y_bo, 0.1)
        X_next = run_bo(
            model=model,
            bounds=bounds,
            train_y=Y_bo,
            ref_point=ref_point,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            device=device
        )
        Y_next = black_box.mechanical_model(X_next)
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)

        # Only non-dominated frontier points
        pareto_mask = is_non_dominated(Y)
        pareto_y = Y[pareto_mask]
        hv = bo_evaluation.get_hyper_volume(pareto_y, ref_point)
        gd = bo_evaluation.get_gd(pareto_y, true_pf)
        igd = bo_evaluation.get_igd(pareto_y, true_pf)
        spacing = bo_evaluation.get_spacing(pareto_y)
        cardinality = bo_evaluation.get_cardinality(pareto_y)
        hv_history.append(hv)
        gd_history.append(gd)
        igd_history.append(igd)
        spacing_history.append(spacing)
        cardinality_history.append(cardinality)
    print(f"\n========= X =========")
    print(X)
    print(f"\n========= Y =========")
    print(Y)
    print(f"\n========= HV History =========")
    print(hv_history)
    pd.DataFrame(X.cpu().numpy()).to_csv("X_all_1.csv", index=False)
    pd.DataFrame(Y.cpu().numpy()).to_csv("Y_all_1.csv", index=False)
    pd.DataFrame({"hyper_volume": hv_history}).to_csv("hv_1.csv", index=False)
    plt.plot(hv_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over BO iterations")
    plt.grid(True)
    plt.savefig("hv_plot_1.png")


pass
if __name__ == "__main__":
    main()
