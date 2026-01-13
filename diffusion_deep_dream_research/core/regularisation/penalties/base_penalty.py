from pydantic import BaseModel
from abc import ABC, abstractmethod
import torch



class BasePenalty(BaseModel, ABC):
    """
    Base class for all penalties.
    """
    weight: float


    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        penalty = self.compute_penalty(latents)
        return self.weight * penalty

    @abstractmethod
    def compute_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        pass