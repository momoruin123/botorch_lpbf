import pandas as pd
import torch
from botorch.utils import draw_sobol_samples

from models import black_box
from optimization import qLogEHVI, qEI


def read_data(x_dim, y_dim, file_path: str, x_cols=None, y_cols=None):
    """
    Read training data from .csv. If no column name is given, the first column is taken by default:
        input_dim column as X,
        objective_dim column as Y.
    :param x_dim: dimension of X
    :param y_dim: dimension of Y
    :param file_path: file path
    :param x_cols: column names of training data
    :param y_cols: column names of training data
    :return: X, Y
    """
    df = pd.read_csv(file_path)

    if x_cols is None:
        x_cols = df.columns[:x_dim]
    if y_cols is None:
        y_cols = df.columns[x_dim:x_dim + y_dim]
    X_np = df[list(x_cols)].to_numpy()
    Y_np = df[list(y_cols)].to_numpy()
    X = torch.as_tensor(X_np)
    Y = torch.as_tensor(Y_np)
    return X, Y


def generate_initial_data(M, bounds: torch.Tensor, n_init: int, device: torch.device) -> tuple:

    if n_init == 0:
        sobol_x = torch.zeros((n_init, 10), device=device)
        S = torch.zeros((n_init, 2), device=device)
        Q = torch.zeros((n_init, 2), device=device)
        return sobol_x, S, Q

    sobol_P = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1).to(device)
    M = torch.tensor(M, device=device).repeat(n_init, 1)
    sobol_x = torch.cat((M, sobol_P), dim=-1)  # x = [M, P]
    # for using
    S, Q = black_box.PMSQ_model(sobol_x)
    return sobol_x, S, Q


def run_singletask_bo(
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


def run_multitask_bo(
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
        model (ModelListGP): Trained multi-objective GP models.
        bounds (torch.Tensor): Optimization variable bounds [2, d].
        train_y (torch.Tensor): Training objectives, shape [N, 2].
        ref_point (list): Reference point in objective space, e.g., [0.5, 0.5].
        batch_size (int): Target number of new samples to generate.
        mini_batch_size (int): BO internal batch size per iteration.
        device (torch.device): Target device (CPU/GPU).

    Returns:
        torch.Tensor: New candidate points, shape [batch_size, d].
    """
    X_next_tensor = torch.empty((0, bounds.shape[1]), device=device)
    iteration = 0

    while X_next_tensor.shape[0] < batch_size:
        X_candidates, acq_val = qLogEHVI.optimize_acq_fun(
            model=model,
            train_y=train_y,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=mini_batch_size,
            num_restarts=10,
            raw_samples=128,
        )
        X_next_tensor = torch.cat((X_next_tensor, X_candidates), dim=0)
        iteration += 1
        # print(f"[BO] Iter {iteration}: Added {X_candidates.shape[0]} → total {X_next_tensor.shape[0]}")
    return X_next_tensor[:batch_size, :]


def attach_feature_vector(x: torch, v: list):
    """
    Attach feature vectors to x
    :param x: input set, shape (n_samples, n_features)
    :param v: feature vectors, shape (1, n_embedding_features)
    :return: augmented set, shape (n_samples, n_features + n_embedding_features)
    """
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.double, device=x.device)

    v = v.repeat(x.shape[0], 1)
    x_aug = torch.cat((x, v), dim=1)
    return x_aug
