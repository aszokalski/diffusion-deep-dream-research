from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict
import torch.nn as nn

class BaseHook(ABC, BaseModel):
    """Abstract base class for pytorch hooks.
    Hooks can be CaptureHooks or SteeringHooks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def __call__(self, module: nn.Module, input: Any, output: Any):
        pass

class EarlyExit(Exception):
    """
    Exception used to stop the execution of the model.
    """
    pass

@contextmanager
def hook_context(target: nn.Module, hook: BaseHook):
    handle = target.register_forward_hook(hook)
    try:
        yield hook
    except EarlyExit:
        pass
    finally:
        handle.remove()

def create_target_hook_context(target: nn.Module):
    return lambda hook: hook_context(target, hook)