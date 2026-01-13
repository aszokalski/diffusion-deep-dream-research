from dataclasses import dataclass
import torchvision.transforms.functional as TF
import torch
import random


@dataclass
class LatentAugmenter:
    """
    Transformation robustness augmenter for latents.
    """
    jitter_max: int = 0
    rotate_max: float = 0.0
    scale_max: float = 1.0

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        if self.jitter_max > 0:
            ox = random.randint(-self.jitter_max, self.jitter_max)
            oy = random.randint(-self.jitter_max, self.jitter_max)
            latents = torch.roll(latents, shifts=(ox, oy), dims=(2, 3))

        if self.rotate_max > 0 or self.scale_max != 1.0:
            angle = random.uniform(-self.rotate_max, self.rotate_max)
            scale = random.uniform(1.0, self.scale_max)
            latents = TF.affine(
                latents, angle=angle, translate=[0, 0], scale=scale, shear=0,
                interpolation=TF.InterpolationMode.BILINEAR
            )
        return latents
