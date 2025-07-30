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
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from botorch.models import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
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
    Y = black_box.mechanical_model_1(sobol_X)
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
    # torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)
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
    batch_size = 15  # the finial batch size
    mini_batch_size = 15  # If computer is not performing well (smaller than batch_size)
    n_iter = 20  # iterations

    # 0.3 get best value
    X_ref, Y_ref = generate_initial_data(bounds=bounds, n_init=1000, device=device)
    # Bounds of normalization
    y_mecha_min = Y_ref[:, 2:5].min(0).values.to(device)
    y_mecha_max = Y_ref[:, 2:5].max(0).values.to(device)

    f_ref_mecha_n = normalize_static(Y_ref[:, 2:5], y_mecha_min, y_mecha_max)
    f_ref_mecha = objective(f_ref_mecha_n, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)

    # best value
    bv = 1
    print(bv)
    # -------------------- 1. Initial Samples  -------------------- #
    # n_init = 50  # initial samples
    # X, Y = generate_initial_data(bounds=bounds, n_init=n_init, device=device)
    X = torch.tensor([[4.9819e+01, 3.9056e-01, 1.3713e+02],
                      [2.3676e+02, 2.2315e-01, 2.0105e+02],
                      [1.9924e+02, 4.8287e-01, 3.6274e+01],
                      [1.4535e+02, 3.1833e-01, 2.3338e+02],
                      [1.1722e+02, 5.5959e-01, 1.7106e+02],
                      [1.7098e+02, 2.2621e-01, 1.0713e+02],
                      [2.6664e+02, 4.6637e-01, 2.7197e+02],
                      [7.9562e+01, 1.3194e-01, 7.4861e+01],
                      [7.3856e+01, 5.1691e-01, 2.9973e+02],
                      [2.9920e+02, 3.4751e-01, 8.1113e+01],
                      [1.8863e+02, 3.6111e-01, 1.9433e+02],
                      [1.0493e+02, 1.9081e-01, 1.1753e+02],
                      [1.4501e+02, 4.3380e-01, 4.2359e+01],
                      [2.2858e+02, 1.0227e-01, 2.6097e+02],
                      [2.5979e+02, 5.8855e-01, 1.4770e+02],
                      [3.4310e+01, 2.6002e-01, 2.2449e+02],
                      [2.9017e+01, 5.7196e-01, 8.6719e+01],
                      [2.5008e+02, 2.7660e-01, 2.8600e+02],
                      [2.2113e+02, 4.1520e-01, 1.2320e+02],
                      [1.3316e+02, 1.2087e-01, 1.8067e+02],
                      [9.5360e+01, 3.7776e-01, 2.5583e+02],
                      [1.8320e+02, 1.7416e-01, 5.6556e+01],
                      [2.8747e+02, 5.3545e-01, 2.1942e+02],
                      [6.6279e+01, 3.2898e-01, 1.6196e+02],
                      [8.7160e+01, 4.4389e-01, 2.1263e+02],
                      [2.7841e+02, 1.5442e-01, 1.3367e+02],
                      [1.7642e+02, 5.3913e-01, 2.4490e+02],
                      [1.2685e+02, 2.4667e-01, 3.2747e+01],
                      [1.5726e+02, 5.0529e-01, 9.5081e+01],
                      [2.0669e+02, 2.9592e-01, 1.7404e+02],
                      [2.4652e+02, 4.1108e-01, 6.2742e+01],
                      [5.5133e+01, 2.0262e-01, 2.7489e+02],
                      [5.1407e+01, 4.8191e-01, 1.8966e+02],
                      [2.4273e+02, 3.0367e-01, 1.1394e+02],
                      [2.1163e+02, 3.8770e-01, 2.9499e+02],
                      [1.6200e+02, 2.1038e-01, 7.7454e+01],
                      [1.2205e+02, 4.6739e-01, 1.5269e+02],
                      [1.7155e+02, 1.4655e-01, 2.2842e+02],
                      [2.8227e+02, 5.6263e-01, 4.7291e+01],
                      [9.0819e+01, 2.3880e-01, 2.6483e+02],
                      [6.2553e+01, 3.5438e-01, 4.0669e+01],
                      [2.8368e+02, 1.8191e-01, 2.3671e+02],
                      [1.8813e+02, 5.1207e-01, 1.4159e+02],
                      [1.0009e+02, 3.3673e-01, 2.0444e+02],
                      [1.2836e+02, 5.9546e-01, 2.6670e+02],
                      [2.1626e+02, 2.6873e-01, 7.0665e+01],
                      [2.5394e+02, 4.3870e-01, 1.6585e+02],
                      [3.2676e+01, 1.1300e-01, 1.0300e+02],
                      [3.8103e+01, 4.2604e-01, 2.4023e+02],
                      [2.6352e+02, 1.2565e-01, 2.9158e+01]], dtype=torch.float64, device=device)
    Y = torch.tensor([[8.0000e+00, 7.0000e+00, 1.1669e+03, 5.3746e+01, 2.0000e+00, 6.9723e-01],
                      [5.0000e+00, 6.0000e+00, 1.3354e+03, 5.5044e+01, 3.6393e+00, 6.2161e-01],
                      [-0.0000e+00, 0.0000e+00, 1.1998e+03, 4.9566e+01, 2.0335e+00, 7.0683e-01],
                      [2.0000e+00, 3.0000e+00, 1.2914e+03, 4.2195e+01, 2.1347e+00, 8.1795e-01],
                      [8.0000e+00, 6.0000e+00, 1.1950e+03, 3.8261e+01, 2.3386e+00, 7.1685e-01],
                      [3.0000e+00, 2.0000e+00, 1.4660e+03, 4.4651e+01, 2.1757e+00, 5.8684e-01],
                      [1.0000e+00, 1.0000e+00, 1.2591e+03, 5.1857e+01, 2.0153e+00, 5.7032e-01],
                      [-0.0000e+00, 1.0000e+00, 1.2624e+03, 4.3252e+01, 2.0032e+00, 7.2338e-01],
                      [0.0000e+00, 2.0000e+00, 1.2466e+03, 3.5949e+01, 2.0000e+00, 7.2985e-01],
                      [1.0000e+00, 0.0000e+00, 1.3414e+03, 5.7006e+01, 2.4211e+00, 6.5727e-01],
                      [4.0000e+00, 7.0000e+00, 1.2945e+03, 4.7936e+01, 2.0000e+00, 6.5575e-01],
                      [3.0000e+00, 3.0000e+00, 1.2613e+03, 4.7520e+01, 2.0069e+00, 6.9910e-01],
                      [2.0000e+00, 0.0000e+00, 1.1468e+03, 5.2415e+01, 2.0000e+00, 7.3929e-01],
                      [-0.0000e+00, 1.0000e+00, 1.2290e+03, 5.5491e+01, 2.1387e+00, 6.0828e-01],
                      [8.0000e+00, 6.0000e+00, 1.2292e+03, 4.6598e+01, 2.0000e+00, 6.1636e-01],
                      [1.0000e+00, 2.0000e+00, 1.1818e+03, 4.3570e+01, 2.0000e+00, 7.6199e-01],
                      [0.0000e+00, 2.0000e+00, 1.2110e+03, 3.8384e+01, 2.0000e+00, 7.4488e-01],
                      [-0.0000e+00, 0.0000e+00, 1.5099e+03, 5.6704e+01, 4.3946e+00, 5.3809e-01],
                      [7.0000e+00, 4.0000e+00, 1.2304e+03, 4.5387e+01, 2.0000e+00, 6.4995e-01],
                      [9.0000e+00, 8.0000e+00, 1.2412e+03, 3.9173e+01, 2.1839e+00, 7.2218e-01],
                      [1.0000e+00, 0.0000e+00, 1.2451e+03, 3.8979e+01, 2.4133e+00, 7.2790e-01],
                      [1.0000e+00, 0.0000e+00, 1.9611e+03, 5.4040e+01, 3.3031e+00, 5.9950e-01],
                      [4.0000e+00, 3.0000e+00, 1.2001e+03, 4.5698e+01, 2.0000e+00, 6.0150e-01],
                      [9.0000e+00, 1.0000e+01, 1.1971e+03, 4.6774e+01, 2.0444e+00, 8.3055e-01],
                      [3.0000e+00, 3.0000e+00, 1.2158e+03, 4.3178e+01, 2.1293e+00, 7.2422e-01],
                      [6.0000e+00, 3.0000e+00, 1.4913e+03, 4.0884e+01, 2.5822e+00, 5.8205e-01],
                      [1.0000e+00, -0.0000e+00, 1.1724e+03, 3.6898e+01, 2.1286e+00, 7.1252e-01],
                      [0.0000e+00, 1.0000e+00, 1.2122e+03, 4.4197e+01, 2.0000e+00, 6.6261e-01],
                      [2.0000e+00, 1.0000e+00, 1.1703e+03, 4.2993e+01, 2.0000e+00, 6.8329e-01],
                      [1.0000e+01, 1.0000e+01, 1.4837e+03, 5.4192e+01, 2.2064e+00, 5.9709e-01],
                      [0.0000e+00, 0.0000e+00, 1.1400e+03, 3.3672e+01, 2.0000e+00, 6.5367e-01],
                      [-0.0000e+00, -0.0000e+00, 1.7637e+03, 4.9118e+01, 2.5720e+00, 6.5399e-01],
                      [7.0000e+00, 7.0000e+00, 1.1785e+03, 5.1260e+01, 2.0000e+00, 7.9875e-01],
                      [4.0000e+00, 3.0000e+00, 1.4828e+03, 4.9836e+01, 2.1022e+00, 5.9871e-01],
                      [-0.0000e+00, 0.0000e+00, 1.4758e+03, 5.3405e+01, 2.0000e+00, 6.4983e-01],
                      [0.0000e+00, -0.0000e+00, 1.3802e+03, 5.1016e+01, 2.0000e+00, 6.6577e-01],
                      [9.0000e+00, 9.0000e+00, 1.1721e+03, 4.5928e+01, 2.0634e+00, 6.7957e-01],
                      [0.0000e+00, 4.0000e+00, 1.3077e+03, 4.3859e+01, 2.0975e+00, 5.8800e-01],
                      [0.0000e+00, -0.0000e+00, 1.1322e+03, 4.6563e+01, 2.0000e+00, 6.6072e-01],
                      [-0.0000e+00, 0.0000e+00, 1.5529e+03, 4.4901e+01, 2.6047e+00, 6.8876e-01],
                      [0.0000e+00, 0.0000e+00, 1.2736e+03, 4.8501e+01, 2.2471e+00, 7.1271e-01],
                      [2.0000e+00, 2.0000e+00, 1.5721e+03, 3.6447e+01, 2.0584e+00, 6.0654e-01],
                      [8.0000e+00, 6.0000e+00, 1.2340e+03, 3.9811e+01, 2.1413e+00, 7.0238e-01],
                      [5.0000e+00, 7.0000e+00, 1.1977e+03, 3.3252e+01, 2.0281e+00, 8.0781e-01],
                      [1.0000e+00, 0.0000e+00, 1.1852e+03, 3.9676e+01, 2.1414e+00, 6.5966e-01],
                      [2.0000e+00, 1.0000e+00, 1.3280e+03, 5.0309e+01, 2.0000e+00, 5.6517e-01],
                      [9.0000e+00, 8.0000e+00, 1.2486e+03, 4.1055e+01, 2.4049e+00, 6.8469e-01],
                      [4.0000e+00, 1.0000e+00, 1.2247e+03, 5.0399e+01, 2.4294e+00, 7.1026e-01],
                      [1.0000e+00, 1.0000e+00, 1.2445e+03, 4.7873e+01, 2.1208e+00, 7.5650e-01],
                      [0.0000e+00, 0.0000e+00, 1.2590e+03, 4.1979e+01, 2.3290e+00, 6.3375e-01]],
                     dtype=torch.float64, device=device)

    for i in range(n_iter):
        print(f"\n========= Iteration {i + 1}/{n_iter} =========")
        # -------------------- 2. Surrogate Model  -------------------- #
        # 2.1 Stacked GP 1
        gp_1_lv = SingleTaskGP_model.build_single_model(X, Y[:, 0:1])
        gp_1_su = SingleTaskGP_model.build_single_model(X, Y[:, 1:2])
        gp_1_list = ModelListGP(gp_1_lv, gp_1_su)
        Y_1_pred_mean, Y_1_pred_var = SingleTaskGP_model.predict_stacked_gp(gp_1_list, X)

        # 2.2 Stacked GP 2
        X_2 = torch.cat((X, Y_1_pred_mean, Y_1_pred_var), dim=1)

        # Build GP 2 (X[7] -> Y[1])
        norm_mecha = normalize_static(Y[:, 2:5], y_mecha_min, y_mecha_max)
        f_mecha = objective(norm_mecha, weight=[0.34, 0.33, 0.33]).unsqueeze(-1)
        gp_2 = SingleTaskGP_model.build_single_model(X_2, f_mecha)
        model = StackedGPModel(gp_1_list, gp_2)
        Y_bo = f_mecha
        X_next = run_bo(
            model=model,
            bounds=bounds,
            train_y=Y_bo,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            device=device
        )
        Y_next = black_box.mechanical_model_1(X_next)
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

    metrics_df = pd.DataFrame({
        "best_so_far": best_so_far,
        "simple_regret": simple_regret,
    })

    iterations = list(range(1, len(best_so_far) + 1))
    plt.figure(figsize=(8, 6))

    # left Y axis
    ax1 = plt.gca()
    ax1.plot(iterations, best_so_far, marker='o', label='best_so_far')
    ax1.plot(iterations, simple_regret, marker='s', label='simple_regret')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Metric Value (normalized)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    plt.title(f"v3 batch_size={batch_size} n_iter={n_iter}")
    plt.tight_layout()

    # save_dir = '/content/drive/MyDrive'
    # pd.DataFrame(X.cpu().numpy()).to_csv(f"{save_dir}/X_all_{timestamp}.csv", index=False)
    # pd.DataFrame(Y.cpu().numpy()).to_csv(f"{save_dir}/Y_all_{timestamp}.csv", index=False)
    # metrics_df.to_csv(f"{save_dir}/metrics_value_{timestamp}.csv", index=False)
    # plt.savefig(f"{save_dir}/metrics_value_{timestamp}.png")
    pd.DataFrame(X.cpu().numpy()).to_csv(f"X_all_{timestamp}.csv", index=False)
    pd.DataFrame(Y.cpu().numpy()).to_csv(f"Y_all_{timestamp}.csv", index=False)
    metrics_df.to_csv(f"metrics_value_{timestamp}.csv", index=False)
    plt.savefig(f"metrics_value_{timestamp}.png")

    plt.close()


pass
if __name__ == "__main__":
    main()
