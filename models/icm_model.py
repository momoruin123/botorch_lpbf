import torch
from botorch.models import MultiTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


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
    def _prepare_data(x_s, x_t, y_s, y_t, task_id_s, task_id_t):
        # Append task IDs
        xs = torch.cat([x_s, torch.full((x_s.shape[0], 1), task_id_s)], dim=1)
        xt = torch.cat([x_t, torch.full((x_t.shape[0], 1), task_id_t)], dim=1)
        x_all = torch.cat([xs, xt], dim=0)

        ys = y_s.unsqueeze(-1) if y_s.ndim == 1 else y_s
        yt = y_t.unsqueeze(-1) if y_t.ndim == 1 else y_t
        y_all = torch.cat([ys, yt], dim=0)

        return x_all, y_all

    input_dim = x_source.shape[1]
    task_dim = 1  # task ID 维度
    total_input_dim = input_dim + task_dim

    # 对每个目标建立一个 MultiTaskGP
    models = []
    for i in range(3):  # 对于 density, roughness, time
        x_all, y_all = _prepare_data(
            x_source, x_target,
            y_source[:, i:i + 1], y_target[:, i:i + 1],
            task_id_s=0.0, task_id_t=1.0
        )

        model = MultiTaskGP(
            train_X=x_all,
            train_Y=y_all,
            task_feature=-1,  # 最后一维是任务 ID
            output_tasks= [0,1],  # 两个任务
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
    with torch.no_grad():
        return [m.posterior(test_x_aug).mean for m in model.models]
