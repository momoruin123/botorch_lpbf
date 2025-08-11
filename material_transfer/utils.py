import torch
from torch import Tensor
from models import black_box
from botorch.utils import draw_sobol_samples
from optimization import qLogEHVI


def generate_initial_data(model_opt, bounds: torch.Tensor, n_init: int, d: int, device: torch.device) -> tuple:
    """
    use Sobol sequence to generate initial samples in given bounds, and use black_box func to get targets.

    Args:
        model_opt (int) :choose models
        bounds (torch.Tensor): shape [2, d]，Lower and upper
        n_init (int): number of initial samples
        d (int): number of input dimensions
        device (torch.device): Device used for computation

    Returns:
        Tuple of tensors: (X_init, Y_init)
    """
    if n_init == 0:
        sobol_x = torch.zeros((n_init, d), device=device)
        y = torch.zeros((n_init, 2), device=device)
        return sobol_x, y
    sobol_x = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1).to(device)
    # for using
    if model_opt == 0:
        return sobol_x, None
    # for evaluation
    if model_opt == 1:
        y = black_box.transfer_model_1(sobol_x, d)
    elif model_opt == 2:
        y = black_box.transfer_model_2(sobol_x, d)
    else:
        raise ValueError("model_opt must be 1 or 2")
    return sobol_x, y


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
    X_next_tensor = torch.empty((0, bounds.shape[1])).to(device)
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