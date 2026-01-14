import torch

from diffusion_deep_dream_research.core.regularisation.penalties.base_penalty import BasePenalty


class MomentPenalty(BasePenalty):
    """
    Moment Penalty.
    Keeps the image in the standard normal distribution (Mean=0, Std=1).
    Prevents the entire image from drifting into a specific color (mean shift)
    or becoming too washed out/contrasty (variance shift).

    Variance shift fights against total variance penalty tendency to blur the image.
    """

    def compute_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        mean_penalty = torch.norm(latents.mean())
        std_penalty = torch.norm(latents.std() - 1.0)

        total_penalty = mean_penalty + std_penalty
        return total_penalty