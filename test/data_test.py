from models import black_box
import torch

torch.manual_seed(42)  # fix random seed

# Set parameters limit（P, v, t, h）
bounds = torch.tensor([
    [50, 200, 0.09, 0.1],  # Upper bounds
    [150, 1000, 0.13, 0.4]  # Lower bounds
], dtype=torch.double)

N_init = 10
X_train = torch.rand(N_init, 4) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.func1(X_train)
print(Y_train.mean())
print(Y_train)

X_train = torch.rand(N_init, 4) * (bounds[1] - bounds[0]) + bounds[0]
Y_train = black_box.func2(X_train)
print(Y_train.mean())
print(Y_train)
