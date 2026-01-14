import torch
import torch.nn.functional as F

from diffusion_deep_dream_research.core.regularisation.penalties.base_penalty import BasePenalty


class RangePenalty(BasePenalty):
    """
    Range Penalty.
    Prevents latents from going out of a specified VAE range.
    """
    threshold: float

    def compute_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Penalizes values that exceed the statistical 'safe zone' of the VAE.
        Instead of hard clamping, we apply a quadratic penalty to outliers.
        Safe zone for Unit Variance is typically [-3, 3].
        """
        magnitude = torch.abs(latents)
        out_of_bounds = F.relu(magnitude - self.threshold)

        return torch.mean(out_of_bounds ** 2)