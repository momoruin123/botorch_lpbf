import pandas as pd
import torch
from botorch.utils import draw_sobol_samples

from models import black_box


def generate_initial_data(M, bounds: torch.Tensor, n_init: int):
    """
    use Sobol sequence to generate initial samples in given bounds, and use black_box func to get targets.
    """
    sobol_P = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=123).squeeze(1)
    M = torch.tensor(M).repeat(n_init, 1)
    sobol_x = torch.cat((M, sobol_P), dim=-1)
    # for using
    S, Q = black_box.PMSQ_model(sobol_x)

    data = torch.cat([sobol_x, S, Q], dim=1)
    df = pd.DataFrame(data.numpy(), columns=None)
    df.to_csv(f"./dataset.csv", index=False)  # 可改路径


P_bounds = torch.tensor([[0., 0., 0., 0.],
                         [1., 1., 1., 1.]], dtype=torch.double)
M_1 = [0.1857, 0.1165, 0.4962, 0.3767, 0.5447, 0.3063]

generate_initial_data(M_1, P_bounds, 100)
