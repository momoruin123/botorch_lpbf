"""
A BO class with embedding-based representations

:author: Maoyurun Mao
:affiliation: Institut für Strahlwerkzeuge (IFSW), University of Stuttgart
:date: 2025-08-18
"""
import torch

from material_transfer.base_bo_class import BaseBO
from material_transfer.utils import run_multitask_bo, run_singletask_bo
from models import SingleTaskGP_model, MultiTaskGP_model


def attach_feature_vector(x: torch, v: list):
    """
    Attach feature vectors to x
    :param x: input set, shape (n_samples, n_features)
    :param v: feature vectors, shape (1, n_embedding_features)
    :return: augmented set, shape (n_samples, n_features + n_embedding_features)
    """
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.double, device=x.device)

    v = v.repeat(x.shape[0], 1)
    x_aug = torch.cat((x, v), dim=1)
    return x_aug


class EmbeddingBO(BaseBO):
    def __init__(self, input_dim, objective_dim, vector_dim, seed=None, device=None):
        """
        :param input_dim: the dimension of input data, e.g. processing parameters(h, v,...)
        :param objective_dim: the dimension of objective function, e.g. final note of product
        :param vector_dim: the dimension of featrue vectores, e.g. number of features
        :param seed: seed for random number generator
        :param device: device for computation
        """

        super().__init__(input_dim, objective_dim, seed, device)
        # vector_dim
        self.vector_dim = vector_dim
        # source task X/Y init
        self.X = torch.empty((0, input_dim+vector_dim), device=self.device)
        self.Y = torch.empty((0, objective_dim), device=self.device)

        # Determine objective dimensions
        if objective_dim == 1:
            self._is_single_task = True
        else:
            self._is_single_task = False

    def add_augment_data(self, X_new, Y_new, v):
        """Append new observations"""
        X_new = torch.as_tensor(X_new, dtype=self.X.dtype, device=self.device)
        Y_new = torch.as_tensor(Y_new, dtype=self.Y.dtype, device=self.device)
        assert X_new.shape[0] == Y_new.shape[0], "X and Y batch size(Number of lines) mismatch"
        X_aug = attach_feature_vector(X_new, v)
        self.X = torch.cat([self.X, X_aug], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

    def set_bounds(self, lower, upper):
        """lower/upper: array-like length d"""
        lower = torch.as_tensor(lower, dtype=self.X.dtype, device=self.device).view(-1)
        upper = torch.as_tensor(upper, dtype=self.X.dtype, device=self.device).view(-1)
        assert (lower.numel() == self.input_dim+self.vector_dim and
                upper.numel() == self.input_dim+self.vector_dim), "bounds dim mismatch (input_dim+vector_dim)"
        assert torch.all(upper >= lower), "upper must be > lower"
        self.bounds = torch.stack([lower, upper], dim=0)

    def build_model(self):
        """build and return GP model with old task data. Make sure to call when X_src and Y_src is not empty"""
        assert self.X.numel() > 0 and self.Y.numel() > 0, \
            "X_src/Y_src are empty; add_data() before run_bo()."

        model = SingleTaskGP_model.build_model(self.X, self.Y)  # build GP model
        # if self.X_aug.nelement() == 0:
        #     # if no samples for target task, then use GP model of old task to sample.
        #     model = SingleTaskGP_model.build_model(self.X_src, self.Y_src)  # build GP model
        # else:
        #     # else use MultiTaskGP to learn two tasks at the same time and the relationship between them.
        #     model = MultiTaskGP_model.build_model(self.X_src, self.Y_src, self.X, self.Y)  # build GP model

        self.model = model
        return model

    def run_bo(self):
        self.build_model()
        # set y for BO based on whether there is sample
        if self.X.nelement() == 0:  # if not, use source task samples to do BO
            y_bo = self.Y
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
        print(X_next[:, 0:self.input_dim])
        return X_next

    # def _attach_feature_vector(self):
    #     """
    #     Attach feature vectors to x
    #
    #     :param v_src: feature vectors of source task, shape (1, n_embedding_features)
    #     :param v_trg: feature vectors of target task, shape (1, n_embedding_features)
    #     :return: augmented set, shape (n_samples, n_features + n_embedding_features)
    #     """
    #     v_src = None
    #     v_trg = None
    #
    #     if not torch.is_tensor(self.v_src):
    #         v_src = torch.tensor(self.v_src, device=self.device)
    #     if not torch.is_tensor(self.v_src):
    #         v_trg = torch.tensor(self.v_trg, device=self.device)
    #
    #     bounds_ad = torch.tensor(self.v_trg, device=self.device).repeat(2, 1)
    #     self.bounds = torch.cat((self.bounds, bounds_ad), dim=1)
    #
    #     # augment X and X_src
    #     v_src_aug = v_src.repeat(self.X.shape[0], 1)
    #     v_trg_aug = v_trg.repeat(self.X.shape[0], 1)
    #     self.X_src_aug = torch.cat((self.X, v_src_aug), dim=1)
    #     self.X_aug = torch.cat((self.X, v_trg_aug), dim=1)
    #
    #     return self.X_src_aug, self.X_aug
