import torch
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor


class TaskStandardize(OutcomeTransform):
    def __init__(self, num_tasks: int, outputs: int = 1, eps: float = 1e-8):
        super().__init__()
        self.num_tasks = num_tasks
        self.outputs = outputs
        self.eps = eps

        # 保存每个任务的均值和标准差
        self.register_buffer("means", torch.zeros(num_tasks, outputs))
        self.register_buffer("stds", torch.ones(num_tasks, outputs))

    def forward(
            self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        task_ids = X[:, -1].long()
        Y_out = torch.empty_like(Y)
        Yvar_out = torch.empty_like(Yvar) if Yvar is not None else None

        for t in range(self.num_tasks):
            mask = task_ids == t
            if mask.any():
                mean_t = Y[mask].mean(dim=0)
                std_t = Y[mask].std(dim=0).clamp_min(self.eps)
                self.means.data[t].copy_(mean_t)
                self.stds.data[t].copy_(std_t)

                Y_out[mask] = (Y[mask] - mean_t) / std_t
                if Yvar is not None:
                    Yvar_out[mask] = Yvar[mask] / (std_t ** 2)

        return Y_out, Yvar_out

    def untransform(
            self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if X is None:
            raise RuntimeError("Task IDs are required in X for untransform.")

        task_ids = X[:, -1].long()
        Y_out = torch.empty_like(Y)
        Yvar_out = torch.empty_like(Yvar) if Yvar is not None else None

        for t in range(self.num_tasks):
            mask = task_ids == t
            if mask.any():
                Y_out[mask] = Y[mask] * self.stds[t] + self.means[t]
                if Yvar is not None:
                    Yvar_out[mask] = Yvar[mask] * (self.stds[t] ** 2)

        return Y_out, Yvar_out

    def forward_transform(self, Y: torch.Tensor, task_ids: torch.Tensor):
        """仅对给定任务 ID 做 forward transform"""
        return (Y - self.means[task_ids]) / self.stds[task_ids]

    def untransform_single(self, Y: torch.Tensor, task_ids: torch.Tensor):
        """仅对给定任务 ID 做 untransform"""
        return Y * self.stds[task_ids] + self.means[task_ids]
