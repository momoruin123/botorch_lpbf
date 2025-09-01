import torch

from models import SingleTaskGP_model
from utils import read_data, get_log_hypers_from_modellist_rbf

results_x = {}
results_y = {}

for i in [1, 3, 4, 5]:
    file_path = f"./datasets/{i}_dataset.csv"
    x, y = read_data(5, 2, file_path)
    results_x[i] = x
    results_y[i] = y

num_targets = 2  # 你是2个目标
ell_logs = [[] for _ in range(num_targets)]  # 每个目标一个桶，装 log(lengthscale) 张量，形状 (d,)
out_logs = [[] for _ in range(num_targets)]  # 装 log(outputscale) 标量
noi_logs = [[] for _ in range(num_targets)]  # 装 log(noise) 标量

for tid in [1, 3, 4, 5]:
    X, Y = read_data(5, 2, f"./datasets/{tid}_dataset.csv")
    model = SingleTaskGP_model.build_model(X, Y)  # 返回 ModelListGP，已fit
    print(type(model.models[0].covar_module))
    # 假设你用的是“RBF、无ScaleKernel”的提取函数：
    logL_list, logS_list, logN_list = get_log_hypers_from_modellist_rbf(model)
    # 注意：这三个是“按目标”的列表：长度=num_targets，每个元素是Tensor

    for k in range(num_targets):
        ell_logs[k].append(logL_list[k])  # 这里 append 的是 Tensor，不是 list
        out_logs[k].append(logS_list[k])
        noi_logs[k].append(logN_list[k])

# 现在对每个目标，沿“任务维度”stack再统计均值/方差
stats_per_obj = []
for k in range(num_targets):
    # ell_logs[k] 是若干个形状为 (d,) 的 Tensor
    ell_stack = torch.stack(ell_logs[k], dim=0)  # (num_tasks, d)
    out_stack = torch.stack(out_logs[k], dim=0)  # (num_tasks,)
    noi_stack = torch.stack(noi_logs[k], dim=0)  # (num_tasks,)

    ell_mu = ell_stack.mean(0)
    ell_std = ell_stack.std(0).clamp_min(1e-6)
    out_mu = out_stack.mean()
    out_std = out_stack.std().clamp_min(1e-6)
    noi_mu = noi_stack.mean()
    noi_std = noi_stack.std().clamp_min(1e-6)

    stats_per_obj.append({
        "ell_mu": ell_mu, "ell_std": ell_std,
        "out_mu": out_mu, "out_std": out_std,
        "noi_mu": noi_mu, "noi_std": noi_std,
    })

print("good")
