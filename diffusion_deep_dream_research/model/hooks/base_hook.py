from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
import torch.nn as nn

class BaseHook(ABC, BaseModel):
    """Abstract base class for pytorch hooks.
    Hooks can be CaptureHooks or SteeringHooks.
    """

    @abstractmethod
    def __call__(self, module: nn.Module, input: Any, output: Any):
        pass

