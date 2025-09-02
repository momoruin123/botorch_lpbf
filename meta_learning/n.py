import torch
from torch import nn
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 假设read_data已实现：返回(x, y)，x是(n_samples, 5)的numpy数组，y是(n_samples, 2)的numpy数组
from utils import read_data

# --------------------------
# 1. 读取并处理数据（核心补充部分）
# --------------------------
results = {}

# 源任务ID和对应的粗糙特征向量（2维）
source_tasks = [1, 2, 3, 4, 5]
source_tasks_v = [[1, 0], [0.6, 10.8], [1.2, 13.8], [0.95, 6.5], [0.85, 2.5]]

# 读取所有任务数据并合并特征（原始5维 + 粗糙2维 → 7维）
all_X = []  # 存储所有样本的7维特征
all_y = []  # 存储所有样本的2维标签

for i in range(len(source_tasks)):
    task_id = source_tasks[i]
    file_path = f"./datasets/{task_id}_dataset.csv"
    x, y = read_data(x_dim=5, y_dim=2, file_path=file_path)  # x: (200,5), y: (200,2)

    # 1.1 合并原始特征和粗糙特征（5+2=7维）
    rough_feature = np.array(source_tasks_v[i])  # 当前任务的粗糙特征：(2,)
    # 为该任务的所有样本复制粗糙特征（每个样本都附加相同的粗糙特征）
    x_rough = np.tile(rough_feature, (x.shape[0], 1))  # (200, 2)
    x_extended = np.concatenate([x, x_rough], axis=1)  # (200, 7)

    # 1.2 存入全局列表
    all_X.append(x_extended)
    all_y.append(y)

# 1.3 合并所有任务的数据（4个任务，每个200样本 → 800样本）
all_X = np.concatenate(all_X, axis=0)  # (800, 7)
all_y = np.concatenate(all_y, axis=0)  # (800, 2)

# 1.4 划分训练集（80%）和验证集（20%）
X_train, X_val, y_train, y_val = train_test_split(
    all_X, all_y, test_size=0.2, random_state=42
)

# 1.5 数据标准化（必须做，避免特征尺度影响训练）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 用训练集的均值/标准差标准化
X_val_scaled = scaler.transform(X_val)          # 验证集用同样的规则

# 1.6 转换为PyTorch张量（注意：nn.Linear默认用float32，这里保持一致）
X_train = torch.FloatTensor(X_train_scaled)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val_scaled)
y_val = torch.FloatTensor(y_val)

# --------------------------
# 2. 定义模型（特征提取器 + 任务头）
# --------------------------
# 特征提取器（输入7维：5原始+2粗糙）
extractor = nn.Sequential(
    nn.Linear(7, 32),    # 7→32
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),   # 32→16
    nn.ReLU(),
    nn.Linear(16, 8)     # 16→8（最终共享特征）
)

# 任务头（输出2维）
task_head = nn.Sequential(
    nn.Linear(8, 4),     # 8→4
    nn.ReLU(),
    nn.Linear(4, 2)      # 4→2
)

# --------------------------
# 3. 训练配置
# --------------------------
criterion = nn.MSELoss()  # 回归任务用均方误差
optimizer = optim.Adam(
    list(extractor.parameters()) + list(task_head.parameters()),
    lr=0.001
)

# --------------------------
# 4. 训练模型（修复变量未定义问题）
# --------------------------
epochs = 200
batch_size = 32
n_samples = X_train.shape[0]  # 现在X_train已定义（训练集样本数）

for epoch in range(epochs):
    # 训练模式
    extractor.train()
    task_head.train()
    train_loss = 0.0

    # 手动分批次训练
    for i in range(0, n_samples, batch_size):
        # 获取当前批次（避免索引越界）
        batch_X = X_train[i:min(i+batch_size, n_samples)]
        batch_y = y_train[i:min(i+batch_size, n_samples)]

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        features = extractor(batch_X)
        outputs = task_head(features)  # 输出形状: [batch_size, 2]

        # 计算损失
        loss = criterion(outputs, batch_y)

        # 反向传播和更新
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.shape[0]

    # 计算平均训练损失
    train_loss /= n_samples

    # 验证
    extractor.eval()
    task_head.eval()
    with torch.no_grad():  # 不计算梯度，节省资源
        val_features = extractor(X_val)
        val_outputs = task_head(val_features)
        val_loss = criterion(val_outputs, y_val).item()

    # 打印进度
    if (epoch + 1) % 10 == 0:
        print(f"第{epoch+1}轮: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}")

# --------------------------
# 5. 特征提取示例
# --------------------------
# 取5个验证集样本提取特征
sample_X = X_val[:5]  # 已标准化的7维特征

extractor.eval()
with torch.no_grad():
    extracted_features = extractor(sample_X)

print("\n提取的特征形状:", extracted_features.shape)  # 应为 (5, 8)
print("第二个样本的提取特征:\n", extracted_features.numpy()[1])
print(extracted_features)