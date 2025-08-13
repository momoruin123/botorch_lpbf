"""
A warm start Bayesian Optimization method class.

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-12
"""
import torch

from material_transfer.base_bo_class import BaseBO
from material_transfer.utils import run_bo
from models import SingleTaskGP_model, MultiTaskGP_model


class WarmStartBO(BaseBO):
    def __init__(self, input_dim, objective_dim, mode=None, seed=None, device=None):
        super().__init__(input_dim, objective_dim, seed, device)
        # source task X/Y init
        self.X_src = torch.empty((0, input_dim), device=self.device)
        self.Y_src = torch.empty((0, objective_dim), device=self.device)

        self.warm_start_mode = mode

    def add_source_data(self, X_src, Y_src):
        """Append new observations"""
        X_src = torch.as_tensor(X_src, dtype=self.X_src.dtype, device=self.device).view(-1, self.input_dim)
        Y_src = torch.as_tensor(Y_src, dtype=self.Y_src.dtype, device=self.device).view(-1, self.objective_dim)
        assert X_src.shape[0] == Y_src.shape[0], "X and Y batch size(Number of lines) mismatch"
        self.X_src = torch.cat([self.X_src, X_src], dim=0)
        self.Y_src = torch.cat([self.Y_src, Y_src], dim=0)

    def build_model(self):
        """build and return GP model with old task data. Make sure to call when X_src and Y_src is not empty"""
        assert self.X_src.numel() > 0 and self.Y_src.numel() > 0, \
            "X_src/Y_src are empty; add_data() before build_warm_start_model()."
        if self.X.nelement() == 0:
            # if no samples for target task, then use GP model of old task to sample.
            model = SingleTaskGP_model.build_model(self.X_src, self.Y_src)  # build GP model
        else:
            # else use MultiTaskGP to learn two tasks at the same time and the relationship between them.
            model = MultiTaskGP_model.build_model(self.X_src, self.Y_src, self.X, self.Y)  # build GP model

        self.model = model
        return model

    def run_bo(self):
        self.build_model()
        X_next = run_bo(  # run BO
            model=self.model,
            bounds=self.bounds,
            train_y=self.Y,
            ref_point=self.ref_point,
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            device=self.device
        )
        print(X_next)
        return X_next
