import torch
from botorch.models.model import Model


class StackedGPModel(Model):
    def __init__(self, gp1, gp2):
        super().__init__()
        self.gp1 = gp1  # GP for X → latent
        self.gp2 = gp2  # GP for [X, μ1, σ1²] → target

    @property
    def num_outputs(self) -> int:
        return self.gp2.num_outputs

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        r"""Compute posterior using augmented [X, μ1(X), σ1²(X)] as input to gp2."""
        # Step 1: predict μ1 and σ1² from gp1
        post_gp1 = self.gp1.posterior(X, observation_noise=observation_noise)
        mu1 = post_gp1.mean        # shape (batch_shape, q, d1)
        var1 = post_gp1.variance   # shape (batch_shape, q, d1)

        # Step 2: construct input to gp2: concat [X, μ1, σ1²]
        X_aug = torch.cat([X, mu1, var1], dim=-1)  # shape: (batch, q, d + d1 + d1)

        # Step 3: call gp2 to get posterior
        post_gp2 = self.gp2.posterior(X_aug, output_indices=output_indices,
                                      observation_noise=observation_noise, **kwargs)
        return post_gp2
