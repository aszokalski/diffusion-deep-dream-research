from pydantic import PrivateAttr
import torch
import torchvision.transforms.functional as TF

from diffusion_deep_dream_research.core.regularisation.gradient_transforms.base_gradient_transform import (
    BaseGradientTransform,
)


class GradientSmoother(BaseGradientTransform):
    kernel_size: int
    sigma_start: float
    sigma_end: float
    num_steps: int

    _sigma: float = PrivateAttr(0.0)

    def update_current_step(self, step: int):
        progress = step / max(self.num_steps - 1, 1)
        self._sigma = self.sigma_start + progress * (self.sigma_end - self.sigma_start)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if self._sigma <= 0.0:
            return grad
        # Typing for this function in torchvision is broken, hence the ignore
        return TF.gaussian_blur(grad, kernel_size=self.kernel_size, sigma=self._sigma)  # ty:ignore[invalid-argument-type]
