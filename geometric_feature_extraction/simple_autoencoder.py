import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super(PointNetEncoder, self).__init__()
        self.mlp1 = nn.Linear(3, 64)    # (x,y,z) -> 64
        self.mlp2 = nn.Linear(64, 128)  # 64 -> 128
        self.mlp3 = nn.Linear(128, emb_dim)  # 128 -> emb_dim (e.g. 64)

    def forward(self, x):
        # x: (B, N, 3)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)  # (B, N, emb_dim)

        # max pooling over points
        x = torch.max(x, dim=1)[0]  # (B, emb_dim)
        return x


# ===== 简单的 3D 自编码器 =====
class VoxelAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=32):
        super(VoxelAutoencoder, self).__init__()

        # Encoder: 压缩 3D 体素
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # [B,16,D/2,H/2,W/2]
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # [B,32,D/4,H/4,W/4]
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # [B,64,D/8,H/8,W/8]
            nn.ReLU()
        )

        # bottleneck: 压缩到潜在向量
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, latent_dim)  # 这里假设输入体素大小是 32x32x32

        # Decoder: 从潜在空间重建
        self.fc2 = nn.Linear(latent_dim, 64 * 4 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出范围 [0,1]，适合体素数据
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平成向量
        z = self.fc1(x)  # 潜在向量

        # Decoder
        x = self.fc2(z)
        x = x.view(x.size(0), 64, 4, 4, 4)
        x = self.decoder(x)
        return x, z  # 返回重建结果和特征向量
