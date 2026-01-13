from pydantic import BaseModel
from abc import ABC, abstractmethod
import torch



class BaseGradientTransform(BaseModel, ABC):
    """
    Base class for all gradient transforms.
    """

    @abstractmethod
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        pass