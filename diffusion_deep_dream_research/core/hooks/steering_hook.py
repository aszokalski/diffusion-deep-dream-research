from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from submodules.SAeUron.SAE.sae import Sae
import torch
import torch.nn as nn

from diffusion_deep_dream_research.core.hooks.base_hook import BaseHook
from diffusion_deep_dream_research.core.model.modified_diffusion_pipeline_adapter import (
    ModifiedDiffusionPipelineAdapter,
)
from diffusion_deep_dream_research.utils.torch_utils import (
    reshape_to_batch_spatial_channels,
    restore_from_batch_spatial_channels,
)


class BaseSteeringHook(BaseHook):
    """An abstraction over a PyTorch hook that modifies selected activations during inference.
    """
    timesteps: list[int]
    pipe_adapter: ModifiedDiffusionPipelineAdapter


    @torch.no_grad()
    def __call__(self, module: nn.Module, input: Any, output: Any):
        t = self.pipe_adapter.pipe.unet.current_timestep
        if t not in self.timesteps:
            return output

        if isinstance(output, tuple):
            loc_output = output[0]
        else:
            loc_output = output

        original_shape = loc_output.shape

        loc_output = reshape_to_batch_spatial_channels(module, loc_output)

        activations = self._apply_steering(loc_output, t)

        loc_output = restore_from_batch_spatial_channels(module, activations, original_shape)

        if isinstance(output, tuple):
            output = (loc_output,) + output[1:]
        else:
            output = loc_output

        return output

    def _apply_steering(self, activations: torch.Tensor, timestep: int) -> torch.Tensor:
        # activations: (batch_size, h*w, channels)
        raise NotImplementedError

class VectorSteeringHook(BaseSteeringHook):
    vector: torch.Tensor
    strength: dict[int, float] # timestep -> strength

    @torch.no_grad()
    def _apply_steering(self, activations: torch.Tensor, timestep: int) -> torch.Tensor:
        # (batch_size, h*w, channels) + (channels,)
        return activations + (self.vector * self.strength[timestep])


class ChannelSteeringHook(BaseSteeringHook):
    channel: int
    strength: dict[int, float]

    @torch.no_grad()
    def _apply_steering(self, activations: torch.Tensor, timestep: int) -> torch.Tensor:
        # (batch_size, h*w, channels)
        activations[..., self.channel] += self.strength[timestep]
        return activations


class SteeringHookFactory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sae: Optional[Sae] = None
    pipe_adapter: ModifiedDiffusionPipelineAdapter

    def create(self, *,
               channel: int,
               strength: dict[int, float],
               timesteps: list[int]) -> BaseSteeringHook:

        if self.sae is not None:
            target_device = self.pipe_adapter.pipe.device
            target_dtype = self.pipe_adapter.pipe.unet.dtype

            vector = self.sae.W_dec[channel].detach().clone().to(
                device=target_device, 
                dtype=target_dtype
            )
            
            return VectorSteeringHook(
                vector=vector,
                strength=strength,
                timesteps=timesteps,
                pipe_adapter=self.pipe_adapter
            )
        else:
            return ChannelSteeringHook(
                channel=channel,
                strength=strength,
                timesteps=timesteps,
                pipe_adapter=self.pipe_adapter
            )