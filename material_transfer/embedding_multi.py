import sys, os, warnings

warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from datetime import datetime
from botorch.utils.multi_objective import is_non_dominated
import pandas as pd
import torch
from evaluation import bo_evaluation
from evaluation.printer import print_multi_task_value_metric
from models import SingleTaskGP_model, MultiTaskGP_model
from models import black_box
from utils import generate_initial_data, run_multitask_bo


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


def main():
    # ---------- Config  ---------- #
    # save_dir = '/content/drive/MyDrive'
    save_dir = './result'
    method = "embedding"
    batch_size = 2
    mini_batch_size = 2
    test_iter = 1  # Number of testing
    n_iter = 2  # Number of iterations
    n_init_samples = 0  # Number of initial samples of new task

    # ---------- 0. Initialization  ---------- #
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)  # Fixed random seed to reproduce results (Default: negative)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 0.1 Set constance and Hyper parameters
    d = 5
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(device)  # bounds = [0,1]^d
    # 0.2 Get true pareto frontier
    X_ref, Y_ref = generate_initial_data(2, bounds, 1000, d, device)  # [1000, M]
    mask_ref = is_non_dominated(Y_ref)
    true_pf = Y_ref[mask_ref]  # [P, 2]
    # ref_point = qLogEHVI.get_ref_point(Y_ref, 0.1)  # set reference point
    # ref_point = [-0.5319,  0.2925]  # nonlinear
    ref_point = [10.6221, 11.1111]  # linear

    # ---------- 1. Initial Samples  ---------- #
    X_old, Y_old = generate_initial_data(1, bounds, 100, d, device)
    X_old = attach_feature_vector(X_old, [1, 0])
    X_new_init, Y_new_init = generate_initial_data(2, bounds, n_init_samples, d, device=device)
    v_new = [0.6, 10.8]
    X_new_init = attach_feature_vector(X_new_init, v_new)
    bounds = torch.cat((bounds, torch.tensor([v_new]).repeat(2, 1).to(device)), dim=1)
    # ---------- 2. Bayesian Optimization Main Loop ---------- #
    # Log matrix initialization (test_iter × n_iter)
    hv_history = np.zeros((test_iter, n_iter))  # log of hyper volume
    gd_history = np.zeros((test_iter, n_iter))  # log of generational distance
    igd_history = np.zeros((test_iter, n_iter))  # log of inverted generational distance
    spacing_history = np.zeros((test_iter, n_iter))  # log of spacing_history
    cardinality_history = np.zeros((test_iter, n_iter))  # log of cardinality_history
    X_log = torch.empty((0, 7)).to(device)
    Y_log = torch.empty((0, 2)).to(device)
    for j in range(test_iter):
        X_new = X_new_init
        Y_new = Y_new_init
        print(f"\n========= Test {j + 1}/{test_iter} =========")
        for i in range(n_iter):
            print(f"\n========= Iteration {i + 1}/{n_iter} =========")
            if X_new.nelement() == 0:
                # if no samples for new task, then use GP model of old task.
                model = SingleTaskGP_model.build_model(X_old, Y_old)  # build GP model
                X_next = run_multitask_bo(  # run BO
                    model=model,
                    bounds=bounds,
                    train_y=Y_old,
                    ref_point=ref_point,
                    batch_size=batch_size,
                    mini_batch_size=mini_batch_size,
                    device=device
                )
            else:
                model = MultiTaskGP_model.build_model(X_old, Y_old, X_new, Y_new)  # build GP model
                X_next = run_multitask_bo(  # run BO
                    model=model,
                    bounds=bounds,
                    train_y=Y_new,
                    ref_point=ref_point,
                    batch_size=batch_size,
                    mini_batch_size=mini_batch_size,
                    device=device
                )

            Y_next = black_box.transfer_model_2(X_next[:, 0:5], d)
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

        X_log = torch.cat((X_log, X_new), dim=0)
        Y_log = torch.cat((Y_log, Y_new), dim=0)

    hv_mean = hv_history.mean(axis=0)  # HV_mean for all test
    gd_mean = gd_history.mean(axis=0)
    igd_mean = igd_history.mean(axis=0)
    spacing_mean = spacing_history.mean(axis=0)
    cardinality_mean = cardinality_history.mean(axis=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save raw data
    metrics_df = pd.DataFrame({
        "hyper_volume": hv_mean,
        "gd": gd_mean,
        "igd": igd_mean,
        "spacing": spacing_mean,
        "cardinality": cardinality_mean,
    })
    pd.DataFrame(X_log.cpu().numpy()).to_csv(f"{save_dir}/{timestamp}_{method}_X.csv", index=False)
    pd.DataFrame(Y_log.cpu().numpy()).to_csv(f"{save_dir}/{timestamp}_{method}_Y.csv", index=False)
    metrics_df.to_csv(f"{save_dir}/{timestamp}_{method}_value.csv", index=False)
    # print figure
    print_multi_task_value_metric(
        batch_size, mini_batch_size, test_iter, n_iter, n_init_samples,  # parameters of BO
        hv_mean, gd_mean, igd_mean, spacing_mean, cardinality_mean,  # evaluation value of BO
        method=method,
        timestamp=timestamp,
        save_dir=save_dir,
        limit_axes=[[0, 6], [0, 30]]
    )


if __name__ == "__main__":
    main()
