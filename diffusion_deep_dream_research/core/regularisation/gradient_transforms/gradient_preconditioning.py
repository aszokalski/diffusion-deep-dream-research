from typing import Any

from pydantic import PrivateAttr
import torch

from diffusion_deep_dream_research.core.regularisation.gradient_transforms.base_gradient_transform import (
    BaseGradientTransform,
)


class GradientPreconditioner(BaseGradientTransform):
    """
    Gradient Preconditioning
    It scales the gradient signal by inverse frequency in the Fourier domain.
    This encourages the model to make larger updates to low-frequency components,
    preventing high-frequency noise.
    """

    latent_height: int
    latent_width: int
    device: torch.device

    _scale: torch.Tensor = PrivateAttr(torch.tensor([]))

    def model_post_init(self, context: Any, /) -> None:
        Y, X = torch.meshgrid(
            torch.arange(self.latent_height, device=self.device),
            torch.arange(self.latent_width, device=self.device),
            indexing="ij",
        )

        X = X - self.latent_width // 2
        Y = Y - self.latent_height // 2

        freq_sq = X**2 + Y**2
        freqs = torch.sqrt(freq_sq.float())

        scale_pre_fft = 1.0 / (freqs + 1.0)
        self._scale = torch.fft.ifftshift(scale_pre_fft)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        grad_fft = torch.fft.fft2(grad.float())

        grad_fft_scaled = grad_fft * self._scale.unsqueeze(0).unsqueeze(0)

        grad_decorrelated = torch.fft.ifft2(grad_fft_scaled).real

        norm_factor = grad.std() / (grad_decorrelated.std() + 1e-8)  # prevent division by zero

        return grad_decorrelated.to(grad.dtype) * norm_factor
