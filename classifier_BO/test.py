#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
import torch
#%%
def plot_fused_decision_boundary(x, y, objective, new_candidates = None) -> None:
    x_min, x_max = x[:, 0].min() - 50, x[:, 0].max() + 50
    y_min, y_max = x[:, 1].min() - 0.25, x[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    mask = objective >= 8.5

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.3, cmap='coolwarm_r')
    plt.scatter(x[:, 0], x[:, 1], c=mask, edgecolor='k', cmap = ListedColormap(['lightgray', 'green']))
    if new_candidates is not None:
        plt.scatter(new_candidates[:, 0], new_candidates[:, 1], c='black', marker='x', s=100, label='New Candidates')
    plt.xlabel('Power')
    plt.ylabel('Hatch Distance')
    plt.title('Fused Classification Decision Boundary (Random Forest)')
    red_patch = patches.Patch(color=plt.cm.coolwarm_r(1.0), label='Fused')  # red area
    blue_patch = patches.Patch(color=plt.cm.coolwarm_r(0.0), label='Not Fused')  # blue area
    green_patch = patches.Patch(facecolor='green', edgecolor='k', label='objective ≥ 8.5')
    gray_patch = patches.Patch(facecolor='lightgray', edgecolor='k', label='objective < 8.5')
    black_patch = patches.Patch(facecolor='black', edgecolor='k', label='New Candidates')

    plt.legend(handles=[red_patch, blue_patch, green_patch, gray_patch, black_patch], loc='lower right')
    plt.grid(True)
    plt.show()


#%%
# ---------- 0. Initialization  ---------- #
# matplotlib.use("TkAgg")  # Fix compatibility issues between matplotlib and botorch
# torch.manual_seed(42)   # Fixed random seed to reproduce results (Default: negative)

# 0.1 Set constance and Hyper parameters
# Set parameters limit（Power, Hatch_distance）
bounds = torch.tensor([
    [25, 0.1, 25],  # Lower bounds
    [300, 0.6, 300]  # Upper bounds
], dtype=torch.double)

# ---------- 1. Initial Samples  ---------- #
# Initial Samples from old tasks
df = pd.read_csv("RF_batch_bo_summary.csv")
X = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double)
Y = torch.tensor(df[["label_visibility", "surface_uniformity"]].values, dtype=torch.double)
Fused_Label = torch.tensor(df[["fused"]].values, dtype=torch.double).squeeze()
Objective = torch.tensor(df[["objective"]].values, dtype=torch.double).squeeze()
df = pd.read_csv("Candidates.csv")
X_candidates = torch.tensor(df[["power", "hatch_distance", "outline_power"]].values, dtype=torch.double)

# Y = Y[:, 1:]  # only need the last three columns
# print(X.shape, Y.shape)

# ---------- 2. Random Forest Classifier  ---------- #
X_RF = X[:,0:2]
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
print("=== report ===")
print(classification_report(y_RF_test, y_pred))

# 2.4. visualization
plot_fused_decision_boundary(X_RF, Fused_Label, Objective, X_candidates)
#%%

#%%
