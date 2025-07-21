import torch
import numpy as np
from botorch.test_functions import Hartmann, Ackley
from torch import Tensor


def mechanical_model(x: torch.Tensor) -> Tensor:
    """
    Simulated black-box function for LPBF laser process.

    Args:
        x (np.ndarray): shape [N, 3], where columns are:
            - power (W) in [25, 300]
            - hatch_distance (mm) in [0.1, 0.6]
            - outline_power (W) in [25, 300]

    Returns:
        y (np.ndarray): shape [N, 6], columns are:
            - label_visibility [0,1]
            - surface_uniformity [0,1]
            - Young's_modulus (MPa)
            - tensile_strength (MPa)
            - Elongation (%)
            - edge_measurement (mm)
    """
    power = x[:, 0]
    hatch = x[:, 1]
    outline = x[:, 2]

    N = x.shape[0]
    noise = lambda scale: scale * torch.randn(N, dtype=x.dtype, device=x.device)
    eff_density = (0.9 * power + 0.1 * outline) / hatch  # 比例你可以根据实际调整

    # --- Simulating surface properties ---
    label_visibility = 10 * torch.exp(-((outline - 165) / 40) ** 2) + noise(1.0)
    label_visibility = torch.clamp(label_visibility.round(), 0, 10)

    base = 10 * torch.exp(-((outline - 170) / 35) ** 2)
    modulation = torch.exp(-8 * (hatch - 0.3) ** 2)  # 越接近 0.3 越稳定
    surface_uniformity = base * modulation + noise(1.0)
    surface_uniformity = torch.clamp(surface_uniformity.round(), 0, 10)

    # --- Simulating mechanical properties ---
    E_peak1 = 900 * torch.exp(-((eff_density - 800) / 150) ** 2)
    E_peak2 = 800 * torch.exp(-((eff_density - 1600) / 200) ** 2)

    E = 1200 + (E_peak1 + E_peak2) + noise(30)
    strength = (
            45
            + 15 * torch.exp(-((eff_density - 1000) / 150) ** 2)
            + 10 * torch.exp(-((eff_density - 1800) / 200) ** 2)
            + 5 * torch.sin(eff_density * 0.01)
            + noise(6)
    )
    elongation = torch.where(
        outline < 150,
        2.0 + 2.5 * torch.exp(-((outline - 100) / 30) ** 2),
        2.0 + 2.5 * torch.exp(-((outline - 250) / 30) ** 2)
    ) + noise(0.2)

    # --- Marginal error (the smaller, the better) ---
    edge_error = 0.6 - 0.0005 * power + 0.1 * hatch + noise(0.2)
    edge_error = torch.clamp(edge_error, min=0.0)

    # output
    y = torch.stack([
        label_visibility,
        surface_uniformity,
        E,
        strength,
        elongation,
        edge_error
    ], dim=1)

    return y


def func_a(x):
    """
    Build a synthetic black box function.

    :param x: Tasks value
    :return: targets value
    """
    p, v, t, h = x.unbind(dim=1)
    # Volumetric energy density
    ev = p / (v * h * t)

    # Molten pool width & depth (simplified physical empirical formula)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))

    # Target 1: Density ~ sigmoid(ev) and minus micropores caused by too wide a melt pool
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.1))) * torch.exp(-0.05 * (w - d).abs())

    # Target 2: Roughness ~ increases with w and h
    y2 = 5.0 + 0.5 * w + 2.0 * h + 0.05 * (p / v)

    # Goal 3: Processing time ~ is inversely proportional to speed t, but positively correlated with thickness
    y3 = (1 / v + 2 * t + 1 * h) * 10

    return torch.stack([y1, -y2, -y3], dim=-1)


def func_b(x):
    p, v, t, h = x.unbind(dim=1)
    ev = p / (v * h * t)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.12))) * torch.exp(-0.04 * (w - d).abs())
    y2 = 6.0 + 0.4 * w + 2.2 * h + 0.06 * (p / v)
    y3 = (1 / v + 2 * t + 2 * h) * 10
    return torch.stack([y1, -y2, -y3], dim=-1)


# Source task black box function
def func_1(x):
    """
    Build a synthetic black box function.

    :param x: Tasks value
    :return: targets value
    """
    p, v, t, h = x.unbind(dim=1)
    # Volumetric energy density
    ev = p / (v * h * t)

    # Molten pool width & depth (simplified physical empirical formula)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))

    # Target 1: Density ~ sigmoid(ev) and minus micropores caused by too wide a melt pool
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.08))) * torch.exp(-0.06 * (w - d).abs())

    # Target 2: Roughness ~ increases with w and h
    y2 = 5.0 + 0.6 * w + 1.8 * h + 0.04 * (p / v)

    # Goal 3: Processing time ~ is inversely proportional to speed t, but positively correlated with thickness
    y3 = (1 / v + 1.8 * t + 0.4 / h) * 5

    return torch.stack([y1, -y2, -y3], dim=-1)


# Target task black box function
def func_2(x):
    p, v, t, h = x.unbind(dim=1)
    ev = p / (v * h * t)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.12))) * torch.exp(-0.04 * (w - d).abs())
    y2 = 6.0 + 0.4 * w + 2.2 * h + 0.06 * (p / v)
    y3 = (1 / v + 2.0 * t + 0.5 / h) * 5
    return torch.stack([y1, -y2, -y3], dim=-1)
