import torch
import numpy as np
from botorch.test_functions import Hartmann, Ackley


def mechanical_model(x: torch.Tensor) -> np.ndarray:
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
    # --- Simulating surface properties ---
    label_visibility = (10 * (1 - 0.005 * torch.abs(power - 150)) + noise(1.0)).round()
    label_visibility = torch.clamp(label_visibility, 0, 10)

    surface_uniformity = (10 * torch.exp(-3 * (hatch - 0.3) ** 2) + noise(1.0)).round()
    surface_uniformity = torch.clamp(surface_uniformity, 0, 10)

    # --- Simulating mechanical properties ---
    E = 1500 + 800 * torch.exp(-((power - 200) / 80) ** 2) * (1 - hatch) + noise(50)
    strength = 40 + 25 * torch.sin(0.01 * outline) + noise(10)
    elongation = 2 + 3 * torch.exp(-((outline - 150) / 60) ** 2) + noise(0.3)

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
    y3 = (1 / v + 2 * t + 1 * h)*10

    return torch.stack([y1, -y2, -y3], dim=-1)

def func_b(x):
    p, v, t, h = x.unbind(dim=1)
    ev = p / (v * h * t)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.12))) * torch.exp(-0.04 * (w - d).abs())
    y2 = 6.0 + 0.4 * w + 2.2 * h + 0.06 * (p / v)
    y3 = (1 / v + 2 * t + 2 * h)*10
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