from matplotlib import pyplot as plt
from datetime import datetime


def print_multi_task_value_metric(
        batch_size, mini_batch_size, test_iter, n_iter, n_init_samples,  # parameters of BO
        hv_mean, gd_mean, igd_mean, spacing_mean, cardinality_mean,  # evaluation value of BO
        method: str,  # BO method in use
        timestamp: str,
        save_dir: str,  # save path
        limit_axes: list = None  # limit of axes
):
    """
    To print evaluation value figure for Multi-Task Bayesian optimization.

    :param batch_size: parameter of BO
    :param mini_batch_size: parameter of BO
    :param test_iter: parameter of BO
    :param n_iter: parameter of BO
    :param n_init_samples: parameter of BO
    :param hv_mean: evaluation value of BO
    :param gd_mean: evaluation value of BO
    :param igd_mean: evaluation value of BO
    :param spacing_mean: evaluation value of BO
    :param cardinality_mean: evaluation value of BO
    :param method: BO method in use
    :param timestamp: timestamp of the file
    :param save_dir: save path
    :param limit_axes:
        Axis limits for y-axes. If None, axes are auto-scaled.
        Format: [[y1_lower, y1_upper], [y2_lower, y2_upper]].
    :return: None
    """
    # Figure
    iterations = list(range(1, n_iter + 1))
    plt.figure(figsize=(8, 6))

    # Left Y axis for the first four value
    ax1 = plt.gca()
    ax1.plot(iterations, hv_mean, marker='o', label='Hypervolume')
    ax1.plot(iterations, gd_mean, marker='s', label='GD')
    ax1.plot(iterations, igd_mean, marker='^', label='IGD')
    ax1.plot(iterations, spacing_mean, marker='d', label='Spacing')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Metric Value")
    ax1.grid(True)

    # Right Y axis for the cardinality_mean
    ax2 = ax1.twinx()
    ax2.plot(iterations, cardinality_mean, marker='x', color='black', label='Cardinality')
    ax2.set_ylabel("Cardinality", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Set limit of Y axes
    if limit_axes is not None:
        ax1.set_ylim(limit_axes[0][0], limit_axes[0][1])
        ax2.set_ylim(limit_axes[1][0], limit_axes[1][1])

    # Merge legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title("{} BO\n"
              "batch_size = {} mini_batch_size = {} test_iter = {} n_iter = {}\n"
              "n_init_samples = {}".format(method,batch_size, mini_batch_size, test_iter, n_iter, n_init_samples))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{timestamp}_{method}_fig.png")
    plt.close()
