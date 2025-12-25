from typing import Optional, Any, Callable

import torch
from pydantic import PrivateAttr, BaseModel, ConfigDict
import torch.nn as nn

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModel

from diffusion_deep_dream_research.model.hooks.base_hook import BaseHook, EarlyExit
from diffusion_deep_dream_research.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter

from submodules.SAeUron.SAE.sae import Sae


def _reshape_to_batch_spatial_channels(module: nn.Module, activations: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the activations to (batch_size, spatial_dim, channels)
    """
    if isinstance(module, Attention):
        # (batch_size, w*h, channels)
        reshaped_activations = activations
    elif isinstance(module, Transformer2DModel):
        # (batch_size, channels, h, w)
        b, c, h, w = activations.shape
        reshaped_activations = activations.permute(0, 2, 3, 1).reshape(b, h * w, c)
    elif isinstance(module, nn.Conv2d):
        # (batch_size, channels, h, w)
        b, c, h, w = activations.shape
        reshaped_activations = activations.permute(0, 2, 3, 1).reshape(b, h * w, c)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}. Size of activations: {activations.shape}")

    return reshaped_activations


class CaptureHook(BaseHook):
    """An abstraction over a PyTorch hook that captures activations from a module.
    It's used to capture activations and expose them through the `get_last_activations()` method
    """
    detach: bool
    early_exit: bool
    timesteps: list[int]
    pipe_adapter: ModifiedDiffusionPipelineAdapter
    activation_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    _activations: dict[int, torch.Tensor] = PrivateAttr(default_factory=dict)

    def __call__(self, module: nn.Module, input: Any, output: Any):
        t = self.pipe_adapter.pipe.unet.current_timestep
        if t not in self.timesteps:
            return output

        if self.detach:
            loc_outputs = output.detach()
        else:
            loc_outputs = output

        if isinstance(loc_outputs, tuple):
            loc_outputs = loc_outputs[0]

        loc_outputs = _reshape_to_batch_spatial_channels(module, loc_outputs)

        self._activations[t] = self.process_activations(loc_outputs)

        if self.early_exit:
            raise EarlyExit()

        return output

    def process_activations(self, activations: torch.Tensor) -> torch.Tensor:
        # activations: (batch_size, h*w, channels)
        transformed_activations = self.activation_transform(
            activations) if self.activation_transform is not None else activations
        # The transform (if specified) transforms the channel dimension of the activations
        return torch.mean(transformed_activations, dim=(0, 1))  # (channels or transformed_channels,)

    def get_last_activations(self) -> dict[int, torch.Tensor]:
        if self._activations is None:
            raise RuntimeError("No activations captured yet. Run the model first.")

        return self._activations

    def clear_activations(self):
        self._activations = {}


class CaptureHookFactory(BaseModel):
    """
    A Factory class for creating CaptureHook objects.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sae: Optional[Sae] = None
    detach: bool
    early_exit: bool
    timesteps: list[int]
    pipe_adapter: ModifiedDiffusionPipelineAdapter

    def create(self) -> CaptureHook:
        if self.sae is not None:
            return CaptureHook(
                detach=self.detach,
                early_exit=self.early_exit,
                timesteps=self.timesteps,
                pipe_adapter=self.pipe_adapter,
                activation_transform=lambda x: self.sae.pre_acts(x),
            )
        else:
            return CaptureHook(
                detach=self.detach,
                early_exit=self.early_exit,
                timesteps=self.timesteps,
                pipe_adapter=self.pipe_adapter
            )
