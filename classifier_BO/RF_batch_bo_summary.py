import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, qLogExpectedHypervolumeImprovement
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
from botorch.models.transforms import Normalize, Standardize

def plot_fused_decision_boundary(x, y, objective, new_candidates = None) -> None:
    x_min, x_max = x[:, 0].min() - 50, x[:, 0].max() + 50
    y_min, y_max = x[:, 1].min() - 0.15, x[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    mask = objective >= 8.5

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.3, cmap='coolwarm_r')
    plt.scatter(x[:, 0], x[:, 1], c=mask, edgecolor='k', cmap = ListedColormap(['lightgray', 'green']))
    if new_candidates is not None:
        plt.scatter(new_candidates[:, 0], new_candidates[:, 1], c='green', marker='x', s=100, label='New Candidates')
    plt.xlabel('Power')
    plt.ylabel('Hatch Distance')
    plt.title('Fused Classification Decision Boundary (Random Forest)')
    red_patch = patches.Patch(color=plt.cm.coolwarm_r(1.0), label='Fused')  # red area
    blue_patch = patches.Patch(color=plt.cm.coolwarm_r(0.0), label='Not Fused')  # blue area
    green_patch = patches.Patch(facecolor='green', edgecolor='k', label='objective ≥ 8.5')
    gray_patch = patches.Patch(facecolor='lightgray', edgecolor='k', label='objective < 8.5')
    plt.legend(handles=[red_patch, blue_patch, green_patch, gray_patch], loc='lower right')
    plt.grid(True)
    plt.show()


def build_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    """
    Build a multi-objective Gaussian Process (GP) surrogate model.

    :param train_x: Input tensor of shape (K, M), where
                    K is the number of samples and
                    M is the number of process parameters (e.g., power, speed, etc.).
    :param train_y: Output tensor of shape (K, N), where
                    N is the number of target metrics (e.g., density, roughness, processing time).
    :return: A fitted ModelListGP object containing independent GPs for each objective.
    """
    input_dim = train_x.shape[1]  # M
    num_targets = train_y.shape[1]  # N

    assert num_targets == 3  # train_Y includes three targets

    # build single model for every target
    model_density = SingleTaskGP(train_x, train_y[:, 0:1],
                                 input_transform=Normalize(d=input_dim),
                                 outcome_transform=Standardize(m=1))

    model_roughness = SingleTaskGP(train_x, train_y[:, 1:2],
                                   input_transform=Normalize(d=input_dim),
                                   outcome_transform=Standardize(m=1))

    model_time = SingleTaskGP(train_x, train_y[:, 2:3],
                              input_transform=Normalize(d=input_dim),
                              outcome_transform=Standardize(m=1))

    # Merge models
    model = ModelListGP(model_density, model_roughness, model_time)
    # fitting
    mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model.models]
    for mll in mlls:
        fit_gpytorch_mll(mll)

    return model


def optimize_acq_fun(model, train_y, bounds, ref_point, batch_size=3):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for multi-objective BO.

    :param model: A fitted ModelListGP model (multi-objective surrogate).
    :param train_y: Current training targets (shape: N x M).
    :param bounds: Limit of the tasks
    :param batch_size: The size of the returned suggestion sample (default: 3).
    :param ref_point: Reference point for hyper-volume calculation (Shape: M, Tensor or list).
    :return: candidate (q x d), acquisition values
    """
    # Determine whether it is a tensor
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds, dtype=torch.double)
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double)

    acq_func = build_acq_fun(model, ref_point, train_y)

    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,  # Repeat optimization times with different starting points (to prevent local optimum)
        raw_samples=128,  # Initial random sample number (used to find initial value)
        return_best_only=True,  # Only return optimal solution
    )
    return candidate, acq_value  # suggested samples and average acq_value


def build_acq_fun(model, ref_point, train_y):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for multi-objective BO.

    :param model: A fitted ModelListGP model (multi-objective surrogate).
    :param ref_point: Reference point for hyper-volume calculation (Tensor or list).
    :param train_y: Current training targets (shape: N x M).
    :return: A qLogEHVI acquisition function object.
    """
    if not torch.is_tensor(ref_point):
        ref_point = torch.tensor(ref_point, dtype=torch.double)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=42)

    objective = IdentityMCMultiOutputObjective()

    y_pareto = train_y

    partitioning = NondominatedPartitioning(
        ref_point=ref_point,
        Y=y_pareto
    )

    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
        objective=objective
    )
    return acq_func


if __name__ == "__main__":
    # ---------- 0. Initialization  ---------- #
    # matplotlib.use("TkAgg")  # Fix compatibility issues between matplotlib and botorch
    # torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)

    # 0.1 Set constance and Hyper parameters
    # Set parameters limit（Power, Hatch_distance）
    bounds = torch.tensor([
        [25, 0.1, 25],  # Lower bounds
        [300, 0.6, 300]  # Upper bounds
    ], dtype=torch.double)

    # 0.2 Set BO parameters
    batch_size = 20
    ref_point = torch.tensor([5,5,7],dtype=torch.double) # reference point for optimization

    # ---------- 1. Initial Samples  ---------- #
    # Initial Samples from old tasks
    df = pd.read_csv("RF_batch_bo_summary.csv")
    X = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double)
    Y = torch.tensor(df[["edge_clarity", "label_visibility", "surface_uniformity"]].values, dtype=torch.double)
    Fused_Label = torch.tensor(df[["fused"]].values, dtype=torch.double).squeeze()
    Objective = torch.tensor(df[["objective"]].values, dtype=torch.double).squeeze()

    # Y = Y[:, 1:]  # only need the last three columns
    # print(X.shape, Y.shape)

    # ---------- 2. Random Forest Classifier  ---------- #
    X_RF = X[:,0:2].cpu().numpy()
    # 2.1 Split test set (test_size: Test set ratio)
    X_RF_train, X_RF_test, y_RF_train, y_RF_test = train_test_split(X_RF, Fused_Label, test_size=0.2, random_state=37)

    # 2.2 Train (n_estimators: # of random trees, the higher, the better)
    clf = RandomForestClassifier(n_estimators=5000, random_state=96)
    clf.fit(X_RF_train, y_RF_train)

    # 2.3 Evaluation
    #   model \ target |    positive     |    negative
    # ----------------------------------------------------
    #      true        |  true_positive  | false_positive
    # ----------------------------------------------------
    #      false       | false_negative  | true_negative
    # ----------------------------------------------------

    # precision = true_positive/(true_positive+false_positive)
    # recall = true_positive/(true_positive+false_negative)
    # F1 Score = 2 * (precision * recall) / (precision + recall)
    y_pred = clf.predict(X_RF_test)
    print("============ GP report ============")
    print(classification_report(y_RF_test, y_pred))

    # 2.4. visualization
    plot_fused_decision_boundary(X_RF, Fused_Label, Objective)

    # ---------- 3. Bayesian Optimization  ---------- #
    model = build_model(X, Y)  # Build GP model
    X_next = np.empty((0, X.shape[1]))
    if len(X_next) < batch_size:
        X_candidates, acq_val = optimize_acq_fun(
            model=model,
            train_y=Y,
            bounds=bounds,
            ref_point=ref_point,
            batch_size=batch_size
        )
        X_candidates = X_candidates.detach().cpu().numpy()

        # filter
        preds = clf.predict(X_candidates)
        fused_mask = preds == 1
        fused_points = X_candidates[fused_mask]

        X_next = np.vstack([X_next, fused_points])
        fused_mask = preds == 1
        X_next = X_next[fused_mask]

    print(X_next)
    plot_fused_decision_boundary(X_RF, Fused_Label, Objective, X_next)  # show candidates