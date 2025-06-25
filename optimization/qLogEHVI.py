from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, qLogExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
import torch


def optimize_acq_fun(model, train_y, bounds, batch_size=3, ref_point=None, slack=None):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for multi-objective BO.

    :param model: A fitted ModelListGP model (multi-objective surrogate).
    :param train_y: Current training targets (shape: N x M).
    :param bounds: Limit of the tasks
    :param batch_size: The size of the returned suggestion sample (default: 3).
    :param ref_point: Reference point for hyper-volume calculation (Shape: M, Tensor or list).
    :param slack: Slack to get ref_point automatically (Shape: M)
    :return: candidate (q x d), acquisition values
    """
    if ref_point is None and slack is None:
        raise ValueError("You must provide either a ref_point or a slack value to compute.")

    if ref_point is None:
        ref_point = get_ref_point(train_y, slack)

    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds, dtype=torch.double)
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double)

    acq_func = build_acq_fun(model, ref_point, train_y)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,  # Repeat optimization times with different starting points (to prevent local optimum)
        raw_samples=100,  # Initial random sample number (used to find initial value)
        return_best_only=True,  # Only return optimal solution
    )
    return candidate, acq_value  # suggested samples and average acq_value


def get_ref_point(train_y, slack):
    """
    Find a reference point for hyper-volume optimization.

    :param train_y: Current training targets (shape: N x M).
    :param slack: Slack to get ref_point automatically (Shape: M)
    :return: A reference point (shape: M).
    """
    ref_point = []
    for i in range(train_y.shape[1]):
        ref = train_y[:, i].max().item() + (slack[i] if isinstance(slack, list) else slack)
        ref_point.append(ref)
    return ref_point


def build_acq_fun(model, ref_point, train_y):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for multi-objective BO.

    :param model: A fitted ModelListGP model (multi-objective surrogate).
    :param ref_point: Reference point for hyper-volume calculation (Tensor or list).
    :param train_y: Current training targets (shape: N x M).
    :return: A qLogEHVI acquisition function object.
    """
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=42)

    objective = IdentityMCMultiOutputObjective()

    y_pareto = train_y

    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=y_pareto
    )

    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
        objective=objective
    )
    return acq_func



