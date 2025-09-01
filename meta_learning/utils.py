import pandas as pd
import torch
from botorch.utils import draw_sobol_samples

from models import black_box


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


def generate_initial_data(model_opt, bounds: torch.Tensor, n_init: int, d: int):
    """
    use Sobol sequence to generate initial samples in given bounds, and use black_box func to get targets.
    """
    if n_init == 0:
        sobol_x = torch.zeros((n_init, d))
        y = torch.zeros((n_init, 2))
        return sobol_x, y
    sobol_x = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1)
    # for using
    if model_opt == 0:
        return sobol_x, None
    # for evaluation
    if model_opt == 1:
        y = black_box.transfer_model_1(sobol_x, d)
    elif model_opt == 2:
        y = black_box.transfer_model_2(sobol_x, d)
    elif model_opt == 3:
        y = black_box.transfer_model_3(sobol_x, d)
    elif model_opt == 4:
        y = black_box.transfer_model_4(sobol_x, d)
    elif model_opt == 5:
        y = black_box.transfer_model_5(sobol_x, d)
    else:
        raise ValueError("model_opt must be 1 to 5")

    data = torch.cat([sobol_x, y], dim=1)
    df = pd.DataFrame(data.numpy(), columns=None)
    df.to_csv(f"./datasets/{model_opt}_dataset.csv", index=False)  # 可改路径


# —— 1) 提取（log 域）：RBFKernel + 无 ScaleKernel ——
def get_log_hypers_from_modellist_rbf(model):
    """
    假设:
      - model 是 ModelListGP
      - 子模型 m.covar_module 是 RBFKernel（没有 ScaleKernel）
      - 噪声可用 m.likelihood.noise 访问
    返回: (logL_list, logS_list, logN_list)
      - logL: (d,)；logS: 标量（恒为0，对应 outputscale=1.0）；logN: 标量
    """
    assert hasattr(model, "models"), "需要传入 ModelListGP"
    logL_list, logS_list, logN_list = [], [], []
    model.eval()
    with torch.no_grad():
        for m in model.models:
            L = m.covar_module.lengthscale.view(-1)  # RBFKernel 直接有 lengthscale
            N = m.likelihood.noise
            logL_list.append(L.log().clone())
            logS_list.append(L.new_tensor(0.0))      # 没有 outputscale -> log(1)=0
            logN_list.append(N.log().clone())
    return logL_list, logS_list, logN_list
