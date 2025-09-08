import torch

from utils import generate_initial_data

bounds = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]], dtype=torch.double)


# generate_initial_data(1, bounds, 100, 5)
generate_initial_data(2, bounds, 100, 5)
# generate_initial_data(3, bounds, 100, 5)
# generate_initial_data(4, bounds, 100, 5)
# generate_initial_data(5, bounds, 100, 5)
