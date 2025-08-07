# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from datetime import datetime
from botorch.utils.multi_objective import is_non_dominated
from evaluation import bo_evaluation
import pandas as pd
import torch
from models import SingleTaskGP_model
from optimization import qLogEHVI
from models import black_box
import matplotlib.pyplot as plt
from warm_start import run_bo, generate_initial_data


def main():
    # ---------- 0. Initialization  ---------- #
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)  # Fixed random seed to reproduce results (Default: negative)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 0.1 Set constance and Hyper parameters
    d = 5
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(device)  # [0,1]^d
    # 0.2 Get true pareto frontier
    X_ref, Y_ref = generate_initial_data(2, bounds, 1000, d, device)  # [1000, M]
    mask_ref = is_non_dominated(Y_ref)
    true_pf = Y_ref[mask_ref]  # [P, 2]
    # ref_point = qLogEHVI.get_ref_point(Y_ref, 0.1)  # set reference point
    # ref_point = [-0.5319,  0.2925]  # nonlinear
    ref_point = [10.6221, 11.1111]  # linear

    # ---------- 1. Initial Samples  ---------- #
    X_new_init, Y_new_init = generate_initial_data(2, bounds, 20, d, device)

    # ---------- 2. Bayesian Optimization Main Loop ---------- #
    batch_size = 2
    mini_batch_size = 2
    test_iter = 1  # Number of testing
    n_iter = 10  # Number of iterations
    # Log matrix initialize (test_iter × n_iter)
    hv_history = np.zeros((test_iter, n_iter))  # log of hyper volume
    gd_history = np.zeros((test_iter, n_iter))  # log of generational distance
    igd_history = np.zeros((test_iter, n_iter))  # log of inverted generational distance
    spacing_history = np.zeros((test_iter, n_iter))  # log of spacing_history
    cardinality_history = np.zeros((test_iter, n_iter))  # log of cardinality_history
    X_log = torch.empty((0, 5)).to(device)
    Y_log = torch.empty((0, 2)).to(device)
    for j in range(test_iter):
        X_new = X_new_init
        Y_new = Y_new_init
        print(f"\n========= Test {j + 1}/{test_iter} =========")
        for i in range(n_iter):
            print(f"\n========= Iteration {i + 1}/{n_iter} =========")
            model = SingleTaskGP_model.build_model(X_new, Y_new)  # build GP model
            Y_bo = Y_new  # merge training set
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

        X_log = torch.cat((X_log, X_new), dim=0)
        Y_log = torch.cat((Y_log, Y_new), dim=0)

    hv_mean = hv_history.mean(axis=0)  # HV_mean for all test
    gd_mean = gd_history.mean(axis=0)
    igd_mean = igd_history.mean(axis=0)
    spacing_mean = spacing_history.mean(axis=0)
    cardinality_mean = cardinality_history.mean(axis=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_df = pd.DataFrame({
        "hyper_volume": hv_mean,
        "gd": gd_mean,
        "igd": igd_mean,
        "spacing": spacing_mean,
        "cardinality": cardinality_mean,
    })
    # Figure
    iterations = list(range(1, n_iter + 1))
    plt.figure(figsize=(8, 6))
    # left Y axis
    ax1 = plt.gca()
    ax1.plot(iterations, hv_mean, marker='o', label='Hypervolume')
    ax1.plot(iterations, gd_mean, marker='s', label='GD')
    ax1.plot(iterations, igd_mean, marker='^', label='IGD')
    ax1.plot(iterations, spacing_mean, marker='d', label='Spacing')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Metric Value")
    ax1.grid(True)
    ax1.set_ylim(0, 10)
    # right Y axis
    ax2 = ax1.twinx()
    ax2.plot(iterations, cardinality_mean, marker='x', color='black', label='Cardinality')
    ax2.set_ylabel("Cardinality", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 30)
    # merge legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title("Cold Start BO\n"
              "batch_size = {} mini_batch_size = {} test_iter = {} n_iter = {}".format(batch_size, mini_batch_size
                                                                                       , test_iter, n_iter))
    plt.tight_layout()
    # save_dir = '/content/drive/MyDrive'
    save_dir = './result'
    pd.DataFrame(X_log.cpu().numpy()).to_csv(f"{save_dir}/{timestamp}_cold_X.csv", index=False)
    pd.DataFrame(Y_log.cpu().numpy()).to_csv(f"{save_dir}/{timestamp}_cold_Y.csv", index=False)
    metrics_df.to_csv(f"{save_dir}/{timestamp}_cold_value.csv", index=False)
    plt.savefig(f"{save_dir}/{timestamp}_cold_fig.png")
    plt.close()


if __name__ == "__main__":
    main()
