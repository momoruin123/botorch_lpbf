import pandas as pd
import torch


def read_benchmark_test_data(
        file_path: str,
):
    df = pd.read_csv(file_path)
    # read parameters of BO
    paras = (
        df[["batch_size", "mini_batch_size", "test_iter", "n_iter", "n_init_samples", "method"]].values)
    # read evaluation values of BO
    values = (
        df[["hyper_volume", "gd", "igd", "spacing", "cardinality"]].values)
    return paras, values