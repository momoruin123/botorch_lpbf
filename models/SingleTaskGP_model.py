import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


def build_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    """
    Build a multi-objective Gaussian Process (GP) surrogate model.

    :param train_x: Input tensor of shape (K, M), where
                    K is the number of samples and
                    M is the number of process parameters (e.g., power, speed, etc.).
    :param train_y: Output tensor of shape (K, N), where
                    N is the number of target metrics (e.g., density, roughness, processing time).
    :return: A fitted ModelListGP object containing independent GPs for each objective.
    """

    input_dim = train_x.shape[1]  # M
    num_targets = train_y.shape[1]  # N

    assert num_targets == 3, "train_Y includes three targets（density, roughness, time）"

    # build single model for every target
    model_density = SingleTaskGP(train_x, train_y[:, 0:1],
                                 input_transform=Normalize(d=input_dim),
                                 outcome_transform=Standardize(m=1))

    model_roughness = SingleTaskGP(train_x, train_y[:, 1:2],
                                   input_transform=Normalize(d=input_dim),
                                   outcome_transform=Standardize(m=1))

    model_time = SingleTaskGP(train_x, train_y[:, 2:3],
                              input_transform=Normalize(d=input_dim),
                              outcome_transform=Standardize(m=1))

    # Merge models
    model = ModelListGP(model_density, model_roughness, model_time)
    # fitting
    mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model.models]
    for mll in mlls:
        fit_gpytorch_mll(mll)

    return model


def build_single_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """
    Build a single-objective Gaussian Process (GP) surrogate model.

    :param train_x: Input tensor of shape (K, M), where
                    K is the number of samples and
                    M is the number of process parameters (e.g., power, speed, etc.).
    :param train_y: Output tensor of shape (K, 1), where
                    "1" means the single-objective
    :return: A fitted SingleTaskGP model.
    """
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)  # Guarantee that shape of y is [N, 1]

    input_dim = train_x.shape[1]  # M

    # build single model for every target
    model = SingleTaskGP(train_x, train_y,
                         input_transform=Normalize(d=input_dim),
                         outcome_transform=Standardize(m=1))
    # build mll
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fitting
    fit_gpytorch_mll(mll)

    return model


def predict(model, test_x):
    model.eval()
    with torch.no_grad():
        return model.posterior(test_x).mean


def predict_stacked_gp(model, test_x):
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(test_x)
        mean = posterior.mean
        # 获取方差信息
        variance = posterior.variance  # 边际方差
        # 或者使用后验协方差矩阵
        # covariance_matrix = posterior.covariance_matrix
        return mean, variance
