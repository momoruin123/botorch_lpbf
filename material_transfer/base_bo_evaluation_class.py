import numpy as np
import torch

from material_transfer.base_bo_class import BaseBO
from material_transfer.utils import generate_initial_data


class BaseBOEvaluation(BaseBO):
    def __init__(self, input_dim, objective_dim, seed=None, device=None):
        super().__init__(input_dim, objective_dim, seed, device)
        # evaluation parameters
        self.test_iter = 1  # Number of tests
        self.n_iter = 2  # Number of iterations
        self.n_init_samples = 0  # Number of initial samples of new task

        # log
        self.X_log = torch.empty((0, self.input_dim), device=self.device)  # save X
        self.Y_log = torch.empty((0, self.objective_dim), device=self.device)  # save Y

        test_iter = self.test_iter
        n_iter = self.n_iter
        self.hyper_volume_log = np.zeros((test_iter, n_iter))
        self.gd_mean_log = np.zeros((test_iter, n_iter))
        self.igd_mean_log = np.zeros((test_iter, n_iter))
        self.spacing_mean_log = np.zeros((test_iter, n_iter))
        self.cardinality_mean_log = np.zeros((test_iter, n_iter))

    def run_evaluation_iter(self):
        for j in range(self.test_iter):
            X = self.X
            Y = self.Y
            print(f"\n========= Test {j + 1}/{self.test_iter} =========")
            for i in range(self.n_iter):
                print(f"\n========= Iteration {i + 1}/{self.n_iter} =========")
                self._run_evaluation(X, Y)

    def _run_evaluation(self, X, Y):
        if X.nelement() == 0:
            # if no samples for new task, then use random sampling.
            X_next, _ = generate_initial_data(2, self.bounds, self.batch_size, self.input_dim, self.device)
        else:
            model = SingleTaskGP_model.build_model(X, Y)  # build GP model
            Y_bo = Y  # merge training set
            X_next = run_bo(  # run BO
                model=model,
                bounds=bounds,
                train_y=Y_bo,
                ref_point=ref_point,
                batch_size=batch_size,
                mini_batch_size=mini_batch_size,
                device=device
            )
        Y_next = black_box.transfer_model_2(X_next, d)
        X_new = torch.cat((X_new, X_next), dim=0)
        Y_new = torch.cat((Y_new, Y_next), dim=0)
        # print("Size of raw candidates: {}".format(Y_next.shape))
        # Filter and get pareto solves
        pareto_mask = is_non_dominated(Y_new)
        pareto_y = Y_new[pareto_mask]
        pareto_y = torch.unique(pareto_y, dim=0)
        print("pareto_y: {}".format(pareto_y.detach()))
        # Evaluation
        hv = bo_evaluation.get_hyper_volume(pareto_y, ref_point)
        gd = bo_evaluation.get_gd(pareto_y, true_pf)
        igd = bo_evaluation.get_igd(pareto_y, true_pf)
        spacing = bo_evaluation.get_spacing(pareto_y)
        cardinality = bo_evaluation.get_cardinality(pareto_y)
        # Log
        hv_history[j, i] = hv
        gd_history[j, i] = gd
        igd_history[j, i] = igd
        spacing_history[j, i] = spacing
        cardinality_history[j, i] = cardinality
