import torch

# Source task black box function
def func1(x):
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
    y3 = (1 / v + 1.8 * t + 0.8 / h) * 9

    return torch.stack([y1, -y2, -y3], dim=-1)


# Target task black box function
def func2(x):
    p, v, t, h = x.unbind(dim=1)
    ev = p / (v * h * t)
    w = torch.sqrt(p * t / v)
    d = torch.sqrt(p / (v * h))
    y1 = 1.0 / (1 + torch.exp(-(ev - 0.12))) * torch.exp(-0.04 * (w - d).abs())
    y2 = 6.0 + 0.4 * w + 2.2 * h + 0.06 * (p / v)
    y3 = (1 / v + 2.2 * t + 1.2 / h) * 11.0
    return torch.stack([y1, -y2, -y3], dim=-1)