import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches




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