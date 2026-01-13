from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
import torch



class BaseGradientTransform(BaseModel, ABC):
    """
    Base class for all gradient transforms.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        pass