import torch

from diffusion_deep_dream_research.core.regularisation.penalties.base_penalty import BasePenalty


class TotalVariationPenalty(BasePenalty):
    """
    Total Variation Penalty.
    Encourages spatial smoothness in the latents and prevents high frequency noise.
    """

    def compute_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Total Variation penalty.
        Vertical and Horizontal gradients ^2 sum
        https://en.wikipedia.org/wiki/Total_variation
        """
        # latents: (batch_size, channels, height, width)

        tv_h = torch.abs(latents[:, :, 1:, :] - latents[:, :, :-1, :]).mean()
        tv_w = torch.abs(latents[:, :, :, 1:] - latents[:, :, :, :-1]).mean()
        return tv_h + tv_w
