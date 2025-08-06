import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
from botorch.utils.multi_objective import is_non_dominated
from evaluation import bo_evaluation
import pandas as pd
import torch
from botorch.utils import draw_sobol_samples
from torch import Tensor
from models import MultiTaskGP_model
from optimization import qLogEHVI
from models import black_box
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="linear_operator.utils.interpolation")
torch.set_default_dtype(torch.float64)
# matplotlib.use("TkAgg")  # Fix compatibility issues between matplotlib and botorch
torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def generate_initial_data(bounds: torch.Tensor, n_init: int, d: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    use Sobol sequence to generate initial samples in given bounds, and use black_box func to get targets.

    Args:
        bounds (torch.Tensor): shape [2, d]，Lower and upper
        n_init (int): number of initial samples
        d (int): number of input dimensions
        device (torch.device): Device used for computation

    Returns:
        Tuple of tensors: (X_init, Y_init)
    """
    sobol_x = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1).to(device)
    y = black_box.transfer_model_2(sobol_x, d)
    return sobol_x, y


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
    X_next_tensor = torch.empty((0, bounds.shape[1])).to(device)
    iteration = 0

    while X_next_tensor.shape[0] < batch_size:
        X_candidates, acq_val = qLogEHVI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=mini_batch_size,
            num_restarts=5,
            raw_samples=64,
        )
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")
    return X_next_tensor[:batch_size, :]


def read_data(filename, x_target: list, y_target: list, device: torch.device) -> tuple[Tensor, Tensor]:
    df = pd.read_csv(filename)
    x = torch.tensor(df[x_target].values).to(device)
    y = torch.tensor(df[y_target].values).to(device)
    return x, y


def main():
    # ---------- 0. Initialization  ---------- #
    # 0.1 Set constance and Hyper parameters
    d = 5
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(device) # [0,1]^10
    hv_history = []  # log of hyper volume
    gd_history = []  # log of generational distance
    igd_history = []  # log of inverted generational distance
    spacing_history = []  # log of spacing_history
    cardinality_history = []  # log of cardinality_history

    # 0.2 Get true pareto frontier
    X_ref, Y_ref = generate_initial_data(bounds, 1000, d, device=device)  # [1000, M]
    mask_ref = is_non_dominated(Y_ref)
    true_pf = Y_ref[mask_ref]  # [P, 2]
    # ---------- 1. Initial Samples  ---------- #
    # X_old, Y_old = generate_initial_data(bounds, 100, d, device=device)
    X_new, Y_new = generate_initial_data(bounds, 20, d, device=device)

    # ---------- 2. Bayesian Optimization Main Loop ---------- #
    batch_size = 10
    mini_batch_size = 5
    n_iter = 10  # 迭代次数
    for i in range(n_iter):
        print(f"\n========= Iteration {i + 1}/{n_iter} =========")

        model = MultiTaskGP_model.build_model(X_old, Y_old, X_new, Y_new)
        ref_point = qLogEHVI.get_ref_point(Y_new, 0.1)
        Y_bo = torch.cat((Y_old, Y_new), dim=0).to(device)
        X_next = run_bo(
            model=model,
            bounds=bounds,
            train_y=Y_bo,
            ref_point=ref_point,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            device=device
        )
        Y_next = black_box.transfer_model_2(X_next, d)
        X_new = torch.cat((X_new, X_next), dim=0)
        Y_new = torch.cat((Y_new, Y_next), dim=0)
        print("Size of raw candidates: {}".format(X_new.shape))
        # Filter and get pareto solves
        pareto_mask = is_non_dominated(Y_new)
        pareto_y = Y_new[pareto_mask]
        print("pareto_y: {}".format(pareto_y.detach()))

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

    metrics_df = pd.DataFrame({
        "hyper_volume": hv_history,
        "gd": gd_history,
        "igd": igd_history,
        "spacing": spacing_history,
        "cardinality": cardinality_history,
    })
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

    plt.title("Warm Start BO")
    plt.tight_layout()

    # save_dir = '/content/drive/MyDrive'
    # pd.DataFrame(X.cpu().numpy()).to_csv(f"{save_dir}/X_all_{timestamp}.csv", index=False)
    # pd.DataFrame(Y.cpu().numpy()).to_csv(f"{save_dir}/Y_all_{timestamp}.csv", index=False)
    # metrics_df.to_csv(f"{save_dir}/metrics_value_{timestamp}.csv", index=False)
    # plt.savefig(f"{save_dir}/metrics_value_{timestamp}.png")
    save_dir = './result'
    pd.DataFrame(X_new.cpu().numpy()).to_csv(f"{save_dir}/X_all_{timestamp}.csv", index=False)
    pd.DataFrame(Y_new.cpu().numpy()).to_csv(f"{save_dir}/Y_all_{timestamp}.csv", index=False)
    metrics_df.to_csv(f"{save_dir}/metrics_value_{timestamp}.csv", index=False)
    plt.savefig(f"{save_dir}/metrics_value_{timestamp}.png")

    plt.close()

if __name__ == "__main__":
    main()

