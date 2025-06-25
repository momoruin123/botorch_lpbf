import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


def build_model(train_x: torch.Tensor, train_y: torch.Tensor):
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
    num_targets = train_y.shape[1]  # 应为3

    assert num_targets == 3, "train_Y 应该包含3个目标（density, roughness, time）"

    # 为每个目标分别构建 GP 模型
    model_density = SingleTaskGP(train_x, train_y[:, 0:1],
                                 input_transform=Normalize(d=input_dim),
                                 outcome_transform=Standardize(m=1))

    model_roughness = SingleTaskGP(train_x, train_y[:, 1:2],
                                   input_transform=Normalize(d=input_dim),
                                   outcome_transform=Standardize(m=1))

    model_time = SingleTaskGP(train_x, train_y[:, 2:3],
                              input_transform=Normalize(d=input_dim),
                              outcome_transform=Standardize(m=1))

    # 合并为多目标模型
    model = ModelListGP(model_density, model_roughness, model_time)

    # 拟合每个目标模型
    mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model.models]
    for mll in mlls:
        fit_gpytorch_mll(mll)

    return model


def predict(model, test_x):
    model.eval()
    with torch.no_grad():
        return [m.posterior(test_x).mean for m in model.models]
