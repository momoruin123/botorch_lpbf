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

    models = []

    for i in range(num_targets):
        model_i = SingleTaskGP(
            train_x,
            train_y[:, i:i+1],
            input_transform=Normalize(d=input_dim),
            outcome_transform=Standardize(m=1)
        )
        models.append(model_i)

    # 构造联合模型
    model = ModelListGP(*models)
    # fitting
    mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model.models]
    for mll in mlls:
        fit_gpytorch_mll(mll)

    return model


def predict(model, test_x):
    model.eval()
    with torch.no_grad():
        return [m.posterior(test_x).mean for m in model.models]
