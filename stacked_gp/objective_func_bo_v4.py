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
v3: 使用stackedGP训练，但是改为单目标
v4: 单纯的单目标
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
from optimization import qEI
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
        X_candidates, acq_val = qEI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
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
    torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    best_so_far = []  # log of Hyper volume
    simple_regret = []  # # log of Hyper volume
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

    # 0.3 get best value
    X_ref, Y_ref = generate_initial_data(bounds=bounds, n_init=1000, device=device)
    # Bounds of normalization
    y_mecha_min = Y_ref[:, 2:5].min(0).values
    y_mecha_max = Y_ref[:, 2:5].max(0).values
    f_ref_mecha_n = normalize_static(Y_ref[:, 2:5], y_mecha_min, y_mecha_max)
    f_ref_mecha = objective(f_ref_mecha_n, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)

    # best value
    bv = max(f_ref_mecha)
    print(bv)
    # -------------------- 1. Initial Samples  -------------------- #
    n_init = 50  # initial samples
    X, Y = generate_initial_data(bounds=bounds, n_init=n_init, device=device)

    n_iter = 20  # iterations
    for i in range(n_iter):
        # -------------------- 2. Surrogate Model  -------------------- #
        # Build GP 2 (X[3] -> Y[1])
        norm_mecha = normalize_static(Y[:, 2:5], y_mecha_min, y_mecha_max)
        f_mecha = objective(norm_mecha, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)
        gp_2 = SingleTaskGP_model.build_single_model(X, f_mecha)
        model = gp_2
        Y_bo = f_mecha
        X_next = run_bo(
            model=model,
            bounds=bounds,
            train_y=Y_bo,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            device=device
        )
        Y_next = black_box.mechanical_model(X_next)
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        # print("Size of raw candidates", X.size())

        # Get objective
        norm_mecha = normalize_static(Y[:, 2:5], y_mecha_min, y_mecha_max)
        f_mecha = objective(norm_mecha, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)
        Y_bo_next = f_mecha
        Y_bo = torch.cat([Y_bo, Y_bo_next], dim=0)
        # print(Y_bo.double())

        # Evaluation
        bsf = max(Y_bo)
        sr = bv - bsf
        # Log
        best_so_far.append(bsf.item())
        simple_regret.append(sr.item())

    # print(f"\n========= X =========")
    # print(X)
    # print(f"\n========= Y =========")
    # print(Y)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save_dir = '/content/drive/MyDrive'
    # pd.DataFrame(X.cpu().numpy()).to_csv(f"{save_dir}/X_all_{timestamp}.csv", index=False)
    # pd.DataFrame(Y.cpu().numpy()).to_csv(f"{save_dir}/Y_all_{timestamp}.csv", index=False)
    pd.DataFrame(X.cpu().numpy()).to_csv(f"X_all_{timestamp}.csv", index=False)
    pd.DataFrame(Y.cpu().numpy()).to_csv(f"Y_all_{timestamp}.csv", index=False)

    metrics_df = pd.DataFrame({
        "best_so_far": best_so_far,
        "simple_regret": simple_regret,
    })
    # metrics_df.to_csv(f"{save_dir}/metrics_value_{timestamp}.csv", index=False)
    metrics_df.to_csv(f"metrics_value_{timestamp}.csv", index=False)
    iterations = list(range(1, len(best_so_far) + 1))
    plt.figure(figsize=(8, 6))

    # left Y axis
    ax1 = plt.gca()
    ax1.plot(iterations, best_so_far, marker='o', label='best_so_far')
    ax1.plot(iterations, simple_regret, marker='s', label='simple_regret')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Metric Value (normalized)")
    ax1.legend()
    ax1.grid(True)
    plt.title("BO Metrics over Iterations (Dual Y-axis)")
    plt.tight_layout()
    # plt.savefig(f"{save_dir}/metrics_value_{timestamp}.png")
    plt.savefig(f"metrics_value_{timestamp}.png")
    plt.close()


pass
if __name__ == "__main__":
    main()
