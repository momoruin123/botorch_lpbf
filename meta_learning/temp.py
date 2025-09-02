import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from models import SingleTaskGP_model  # 你的GP模型构建函数
from utils import read_data           # 你的数据读取函数

# --------------------------
# 1. 初始化存储结构
# --------------------------
results_x, results_y = {}, {}

num_targets = 2  # 单目标
ell_logs = [[] for _ in range(num_targets)]  # 每元素(5,)
out_logs = [[] for _ in range(num_targets)]  # 标量
noi_logs = [[] for _ in range(num_targets)]  # 标量

# --------------------------
# 2. 遍历源任务，提取超参数
# --------------------------
source_tasks = [1, 3, 4, 5]

for task_id in source_tasks:
    file_path = f"./datasets/{task_id}_dataset.csv"
    x, y = read_data(x_dim=5, y_dim=num_targets, file_path=file_path)

    # 可选：确保 dtype 一致
    x = x.double()
    y = y.double()

    results_x[task_id] = x
    results_y[task_id] = y

    # 假设 build_model 内部包含拟合/训练逻辑，并返回 ModelListGP（单目标时也可能是单个模型）
    model = SingleTaskGP_model.build_model(x, y)
    model.eval()

    # 统一拿到“一个可迭代的模型列表”
    if hasattr(model, "models"):  # ModelListGP
        model_iter = model.models
    else:  # 单模型（SingleTaskGP）
        model_iter = [model]

    with torch.no_grad():
        for j in range(num_targets):
            m = model_iter[j]
            # 取 lengthscale（RBF 的 base_kernel 上）
            if hasattr(m.covar_module, "base_kernel"):
                length_scale = m.covar_module.base_kernel.lengthscale.reshape(-1)
            else:
                # 兼容万一用户自定义没有 ScaleKernel 的情况
                length_scale = m.covar_module.lengthscale.reshape(-1)

            # 取 outputscale（ScaleKernel 外层的尺度），若无则置 1
            if hasattr(m.covar_module, "outputscale"):
                outputscale = m.covar_module.outputscale
            else:
                outputscale = torch.tensor(1.0, dtype=length_scale.dtype, device=length_scale.device)

            # 取 noise
            noise = m.likelihood.noise

            # 转 log
            logL = length_scale.log().clone()       # (d,)
            logS = outputscale.log().clone()        # 标量
            logN = noise.log().clone()              # 标量

            ell_logs[j].append(logL)
            out_logs[j].append(logS)
            noi_logs[j].append(logN)

# --------------------------
# 3. 统计超参数的跨任务分布（均值/标准差）
# --------------------------
stats_per_target = []
for k in range(num_targets):
    ell_stack = torch.stack(ell_logs[k], dim=0)  # (num_tasks, d)
    out_stack = torch.stack(out_logs[k], dim=0)  # (num_tasks,)
    noi_stack = torch.stack(noi_logs[k], dim=0)  # (num_tasks,)

    ell_mean = ell_stack.mean(dim=0)                 # (d,)
    ell_std  = ell_stack.std(dim=0).clamp_min(1e-6)  # (d,)

    out_mean = out_stack.mean()                      # 标量
    out_std  = out_stack.std().clamp_min(1e-6)       # 标量

    noi_mean = noi_stack.mean()                      # 标量
    noi_std  = noi_stack.std().clamp_min(1e-6)       # 标量

    stats_per_target.append({
        "ell_mu": ell_mean,
        "ell_std": ell_std,
        "out_mu": out_mean,
        "out_std": out_std,
        "noi_mu": noi_mean,
        "noi_std": noi_std,
    })

# 简短展示
print(f"Collected from {len(source_tasks)} tasks.")
for i in range(num_targets):
    print(f"Target {i} stats:")
    print({k: v for k, v in stats_per_target[i].items()})
