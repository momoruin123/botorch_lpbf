import torch
from botorch.models import MultiTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


# ICM: Intrinsic Coregionalization Model (A methode of Kernel_based Transfer)
def build_model(x_source, y_source, x_target, y_target):
    """
    Build a multitask Gaussian Process (GP) surrogate model.

    :param x_source: X from source task
                    Input tensor of shape (K, M), where
                    K is the number of samples and
                    M is the number of process parameters (e.g., power, speed, etc.).
    :param y_source: Y from source task
                    Output tensor of shape (K, N), where
                    N is the number of target metrics (e.g., density, roughness, processing time).
    :param x_target: X from target task
    :param y_target: Y from target task
    :return: A fitted ModelListGP object containing independent GPs for each objective.
    """
    input_dim = x_source.shape[1]
    target_dim = y_target.shape[1]
    task_dim = 1  # task ID dimension
    total_input_dim = input_dim + task_dim

    # Create a MultiTaskGP for each  task
    models = []
    for i in range(target_dim):  # for [density, roughness, time]
        x_source_p, y_source_p = prepare_data(x_source, y_source[:, i:i + 1], 0)
        x_target_p, y_target_p = prepare_data(x_target, y_target[:, i:i + 1], 1)
        x_all = torch.cat([x_source_p, x_target_p], dim=0)
        y_all = torch.cat([y_source_p, y_target_p], dim=0)

        model = MultiTaskGP(
            train_X=x_all,
            train_Y=y_all,
            task_feature=-1,  # 最后一维是任务 ID
            output_tasks=[0, 1],  # 两个任务
            input_transform=Normalize(d=total_input_dim),
            outcome_transform=Standardize(m=1)
        )
        models.append(model)

    model = ModelListGP(*models)

    # 拟合每个模型
    mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model.models]
    for mll in mlls:
        fit_gpytorch_mll(mll)

    return model


def predict(model, test_x_target):
    """
    Predict on target task (task_id = 1.0)
    """
    test_x_aug = torch.cat([test_x_target, torch.ones(test_x_target.shape[0], 1)], dim=1)
    model.eval()
    with torch.no_grad():  # DO NOT compute gradients
        return [m.posterior(test_x_aug).mean for m in model.models]


def prepare_data(x, y, task_id):
    # Append task IDs
    x_and_id = torch.cat([x, torch.full((x.shape[0], 1), task_id)], dim=1)
    y = y.unsqueeze(-1) if y.ndim == 1 else y

    return x_and_id, y
