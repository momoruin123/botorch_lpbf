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


def plot_fused_decision_boundary(x,y, new_candidates=None, filename="decision_boundary.png") -> None:
    x_min, x_max = x[:, 0].min() - 50, x[:, 0].max() + 50
    y_min, y_max = x[:, 1].min() - 0.15, x[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.3, cmap='coolwarm_r')
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', cmap='coolwarm_r')
    if new_candidates is not None:
        plt.scatter(new_candidates[:, 0], new_candidates[:, 1], c='green', marker='x', s=100, label='New Candidates')
    plt.xlabel('Power')
    plt.ylabel('Hatch Distance')
    plt.title('Fused Classification Decision Boundary (Random Forest)')
    red_patch = patches.Patch(color=plt.cm.coolwarm_r(1.0), label='Fused')  # red area
    blue_patch = patches.Patch(color=plt.cm.coolwarm_r(0.0), label='Not Fused')  # blue area
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # ---------- 0. Initialization  ---------- #
    # matplotlib.use("TkAgg")  # Fix compatibility issues between matplotlib and botorch
    # torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 0.1 Set constance and Hyper parameters
    # Set parameters limit（Power, Hatch_distance）
    bounds = torch.tensor([
        [25, 0.1, 25],  # Lower bounds
        [300, 0.6, 300]  # Upper bounds
    ], dtype=torch.double).to(device)

    # 0.2 Set BO parameters
    batch_size = 20  # the finial batch size
    mini_batch_size = 10  # If computer is not performing well (smaller than batch_size)
    ref_point = torch.tensor([800, 12, 7, -0.15], dtype=torch.double).to(device)  # reference point for optimization

    # ---------- 1. Initial Samples  ---------- #
    # Initial Samples from old tasks
    df = pd.read_csv("../../classifier_BO/data.csv")
    X = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double).to(device)
    target = ["Young's modulus","tensile strength","Elongation","Edge measurement"]
    Y = torch.tensor(df[target].values, dtype=torch.double).to(
        device)
    Y[:, 3] = -Y[:, 3]
    Fused_Label = torch.tensor(df[["fused"]].values, dtype=torch.double).squeeze().cpu().numpy()
    # Objective = torch.tensor(df[["objective"]].values, dtype=torch.double).squeeze().cpu().numpy()

    # Y = Y[:, 1:]  # only need the last three columns
    # print(X.shape, Y.shape)

    # ---------- 2. Random Forest Classifier  ---------- #
    X_RF = X[:, 0:2].cpu().numpy()
    # 2.1 Split test set (test_size: Test set ratio)
    X_RF_train, X_RF_test, y_RF_train, y_RF_test = train_test_split(X_RF, Fused_Label, test_size=0.2, random_state=37)

    # 2.2 Train (n_estimators: # of random trees, the higher, the better)
    clf = RandomForestClassifier(n_estimators=5000, random_state=96)
    clf.fit(X_RF_train, y_RF_train)

    y_pred = clf.predict(X_RF_test)
    print("============ GP report ============")
    print(classification_report(y_RF_test, y_pred))

    # 2.4. visualization
    plot_fused_decision_boundary(X_RF, Fused_Label)

    df = pd.read_csv("Batch5_850_12.7_5_0.15.csv")
    X_next_tensor = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double)

    print(X_next_tensor)
    plot_fused_decision_boundary(X_RF, Fused_Label, X_next_tensor, "Batch5_850_12.7_5_0.15.png")  # show candidates
