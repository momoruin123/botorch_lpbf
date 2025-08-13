"""
A BO class with embedding-based representations

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-13
"""
import torch

from material_transfer.base_bo_class import BaseBO
from material_transfer.utils import run_multitask_bo, run_singletask_bo
from models import SingleTaskGP_model, MultiTaskGP_model


class EmbeddingStartBO(BaseBO):
    def __init__(self, input_dim, objective_dim, v_src, v_trg, seed=None, device=None):
        """
        :param input_dim: the dimension of input data, e.g. processing parameters(h, v,...)
        :param objective_dim: the dimension of objective function, e.g. final note of product
        :param v_src: the feature vector of source task
        :param v_trg: the feature vector of target task
        :param seed: seed for random number generator
        :param device: device for computation
        """

        super().__init__(input_dim, objective_dim, seed, device)
        # source task X/Y init
        self.X_src = torch.empty((0, input_dim), device=self.device)
        self.Y_src = torch.empty((0, objective_dim), device=self.device)
        # feature vectors
        self.v_src = v_src
        self.v_trg = v_trg
        # augment X
        self.X_aug = torch.empty((0, input_dim+len(f_v)), device=self.device)
        self.X_src_aug = torch.empty((0, objective_dim+len(f_v)), device=self.device)

        if objective_dim == 1:
            self._is_single_task = True
        else:
            self._is_single_task = False

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
            "X_src/Y_src are empty; add_data() before run_bo()."
        if self.X_aug.nelement() == 0:
            # if no samples for target task, then use GP model of old task to sample.
            model = SingleTaskGP_model.build_model(self.X_src, self.Y_src)  # build GP model
        else:
            # else use MultiTaskGP to learn two tasks at the same time and the relationship between them.
            model = MultiTaskGP_model.build_model(self.X_src, self.Y_src, self.X, self.Y)  # build GP model

        self.model = model
        return model

    def run_bo(self):
        self.build_model()

        # set y for BO based on whether there is sample
        if self.X.nelement() == 0:  # if not, use source task samples to do BO
            y_bo = self.Y_src
        else:  # if yes, use target task samples to do BO
            y_bo = self.Y

        # determine whether it is single-task optimization
        if self._is_single_task:  # if yes, use qlogEI as asq. func.
            X_next = run_singletask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )
        else:  # if no, use qlogEHVI as asq. func.
            X_next = run_multitask_bo(
                model=self.model,
                bounds=self.bounds,
                train_y=y_bo,
                ref_point=self.ref_point,
                batch_size=self.batch_size,
                mini_batch_size=self.mini_batch_size,
                device=self.device
            )
        print(X_next)
        return X_next

    def attach_feature_vector(self, v_src: list, v_trg: list):
        """
        Attach feature vectors to x
        :param v_src: feature vectors of source task, shape (1, n_embedding_features)
        :param v_trg: feature vectors of target task, shape (1, n_embedding_features)
        :return: augmented set, shape (n_samples, n_features + n_embedding_features)
        """
        if not torch.is_tensor(v_src):
            v_src = torch.tensor(v_src, device=self.device)
        if not torch.is_tensor(v_src):
            v_trg = torch.tensor(v_trg, device=self.device)

        # augment X and X_src
        v_src = v_src.repeat(self.X_src.shape[0], 1)
        v_trg = v_trg.repeat(self.X.shape[0], 1)
        self.X_src_aug = torch.cat((self.X_src, v_src), dim=1)
        self.X_aug = torch.cat((self.X, v_trg), dim=1)

        return self.X_aug, self.X_src_aug
