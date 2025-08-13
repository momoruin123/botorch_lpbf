import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
from evaluation.utils import read_benchmark_test_data
from evaluation.printer import print_multi_task_value_metric

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path = "20250807_110516_warm_value.csv"
parameters, values = read_benchmark_test_data(path)
batch_size, mini_batch_size, test_iter, n_iter, n_init_samples, method = parameters[0]
hv_mean, gd_mean, igd_mean, spacing_mean, cardinality_mean = values.T
print_multi_task_value_metric(
    batch_size, mini_batch_size, test_iter, n_iter, n_init_samples,  # parameters of BO
    hv_mean, gd_mean, igd_mean, spacing_mean, cardinality_mean,  # evaluation value of BO
    method=method,
    timestamp=timestamp,
    save_dir=".",
    limit_axes=[[0, 6], [0, 30]]
)