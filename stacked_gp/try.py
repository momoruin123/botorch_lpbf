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
Date: 22/07/2025
"""

'''
v1: 剔除预测分数部分和RF部分，用于测试比较
v2: 优化图表输出，分开y轴显示，加入时间戳；固定归一化尺度
v3: 预计加入
    ·更换模型顺序，使用stackedGP训练，用于比较和v2的好坏
    
    目前问题，单独构建了StackedGP，但是无法与SingleTaskGP合并进行多目标优化
'''

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from botorch.utils.multi_objective import is_non_dominated
from matplotlib import pyplot as plt
from torch import Tensor
from botorch.models import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from evaluation import bo_evaluation
from models import SingleTaskGP_model
from optimization import qLogEHVI
from models import black_box
from datetime import datetime
from models.stacked_gp import StackedGPModel


def generate_initial_data(bounds: torch.Tensor, n_init: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    use Sobol sequence to generate initial samples in given bounds, and use black_box func to get targets.

    Args:
        bounds (torch.Tensor): shape [2, d]，Lower and upper
        n_init (int): number of initial samples
        device (torch.device): Device used for computation

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


def normalize_static(y: torch.Tensor, y_min: torch.Tensor, y_max: torch.Tensor) -> torch.Tensor:
    return (y - y_min) / (y_max - y_min + 1e-8)


def run_bo(
        model,
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
        model: Trained multi-objective GP model.
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
    batch_size = 4  # the finial batch size
    mini_batch_size = 2  # If computer is not performing well (smaller than batch_size)

    # 0.3 get true Pareto frontier
    X_ref, Y_ref = generate_initial_data(bounds=bounds, n_init=1000, device=device)  # [1000, M]
    # Bounds of normalization
    y_mecha_min = Y_ref[:, 2:5].min(0).values
    y_mecha_max = Y_ref[:, 2:5].max(0).values
    y_surf_min = Y_ref[:, 0:2].min(0).values
    y_surf_max = Y_ref[:, 0:2].max(0).values
    f_ref_mecha_n = normalize_static(Y_ref[:, 2:5], y_mecha_min, y_mecha_max)
    f_ref_surf_n = normalize_static(Y_ref[:, 0:2], y_surf_min, y_surf_max)
    f_ref_mecha = objective(f_ref_mecha_n, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)
    f_ref_surf = objective(f_ref_surf_n, weight=[0.5, 0.5]).unsqueeze(-1)
    # ture pareto_solves
    Y_ref_bo = torch.cat([f_ref_mecha, f_ref_surf], dim=1)  # [1000,2]
    mask_ref = is_non_dominated(Y_ref_bo)
    true_pf = Y_ref_bo[mask_ref]  # [P, 2]

    # -------------------- 1. Initial Samples  -------------------- #
    n_init = 50  # initial samples
    X, Y = generate_initial_data(bounds=bounds, n_init=n_init, device=device)

    n_iter = 1  # iterations
    for i in range(n_iter):
        # -------------------- 2. Surrogate Model  -------------------- #
        # 2.1 Stacked GP 1 (X[3] -> Y_mean[2],Y_var[2])
        gp_1_lv = SingleTaskGP_model.build_single_model(X, Y[:, 0:1])
        gp_1_su = SingleTaskGP_model.build_single_model(X, Y[:, 1:2])
        gp_1_list = ModelListGP(gp_1_lv, gp_1_su)
        Y_1_pred_mean, Y_1_pred_var = SingleTaskGP_model.predict_stacked_gp(gp_1_list, X)

        # 2.2 Stacked GP 2 (X[3] -> f_mecha, f_surface)
        X_2 = torch.cat((X, Y_1_pred_mean, Y_1_pred_var), dim=1)
        # evaluationh
        norm_mecha = normalize_static(Y[:, 2:5], y_mecha_min, y_mecha_max)
        norm_surf = normalize_static(Y[:, 0:2], y_surf_min, y_surf_max)
        f_mecha = objective(norm_mecha, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)
        f_surface = objective(norm_surf, weight=[0.5, 0.5]).unsqueeze(-1)
        # build GP models
        gp_2_mecha = SingleTaskGP_model.build_single_model(X_2, f_mecha)
        gp_2 = StackedGPModel(gp_1_list, gp_2_mecha)
        gp_2_surface = SingleTaskGP_model.build_single_model(X, f_surface)

        print(f"\n========= Iteration {i + 1}/{n_iter} =========")
        # -------------------- 4. Bayesian Optimization  -------------------- #
        model = StackedGPModel(gp_2, gp_2_surface)
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
        print("Size of raw candidates", X.size())

        # Get objective
        norm_mecha = normalize_static(Y[:, 2:5], y_mecha_min, y_mecha_max)
        norm_surf = normalize_static(Y[:, 0:2], y_surf_min, y_surf_max)
        f_mecha = objective(norm_mecha, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)
        f_surface = objective(norm_surf, weight=[0.5, 0.5]).unsqueeze(-1)
        # Filter and get pareto solves
        Y_bo_next = torch.cat((f_mecha, f_surface), dim=1)
        Y_bo = torch.cat([Y_bo, Y_bo_next], dim=0)
        pareto_mask = is_non_dominated(Y_bo)
        pareto_y = Y_bo[pareto_mask]
        print(pareto_y.double())

        # Evaluation
        hv = bo_evaluation.get_hyper_volume(pareto_y, ref_point)
        gd = bo_evaluation.get_gd(pareto_y, true_pf)
        igd = bo_evaluation.get_igd(pareto_y, true_pf)
        spacing = bo_evaluation.get_spacing(pareto_y)
        cardinality = bo_evaluation.get_cardinality(pareto_y)
        # Log
        hv_history.append(hv)
        gd_history.append(gd)
        igd_history.append(igd)
        spacing_history.append(spacing)
        cardinality_history.append(cardinality)

    print(f"\n========= X =========")
    # print(X)
    print(f"\n========= Y =========")
    # print(Y)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = '/content/drive/MyDrive'
    pd.DataFrame(X.cpu().numpy()).to_csv(f"{save_dir}/X_all_{timestamp}.csv", index=False)
    pd.DataFrame(Y.cpu().numpy()).to_csv(f"{save_dir}/Y_all_{timestamp}.csv", index=False)

    metrics_df = pd.DataFrame({
        "hyper_volume": hv_history,
        "gd": gd_history,
        "igd": igd_history,
        "spacing": spacing_history,
        "cardinality": cardinality_history,
    })
    metrics_df.to_csv(f"{save_dir}/metrics_value_{timestamp}.csv", index=False)
    iterations = list(range(1, len(hv_history) + 1))
    plt.figure(figsize=(8, 6))

    # left Y axis
    ax1 = plt.gca()
    ax1.plot(iterations, hv_history, marker='o', label='Hypervolume')
    ax1.plot(iterations, gd_history, marker='s', label='GD')
    ax1.plot(iterations, igd_history, marker='^', label='IGD')
    ax1.plot(iterations, spacing_history, marker='d', label='Spacing')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Metric Value (normalized)")
    ax1.grid(True)

    # right Y axis
    ax2 = ax1.twinx()
    ax2.plot(iterations, cardinality_history, marker='x', color='black', label='Cardinality')
    ax2.set_ylabel("Cardinality", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # merge legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title("BO Metrics over Iterations (Dual Y-axis)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_value_{timestamp}.png")
    plt.close()


pass
if __name__ == "__main__":
    main()
