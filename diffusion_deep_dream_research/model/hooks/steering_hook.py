from typing import Optional, Any, Callable

import torch
from pydantic import BaseModel, ConfigDict
import torch.nn as nn

from diffusion_deep_dream_research.model.hooks.base_hook import BaseHook
from diffusion_deep_dream_research.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter
from diffusion_deep_dream_research.utils.torch_utils import reshape_to_batch_spatial_channels

from submodules.SAeUron.SAE.sae import Sae


class SteeringHook(BaseHook):
    """An abstraction over a PyTorch hook that modifies selected activations during inference.
    """
    channels: list[int]
    strength: torch.Tensor
    timesteps: list[int]
    pipe_adapter: ModifiedDiffusionPipelineAdapter
    activation_encode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    activation_decode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


    @torch.no_grad()
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

        original_shape = loc_outputs.shape

        loc_outputs = reshape_to_batch_spatial_channels(module, loc_outputs)

        activations = self.process_activations(loc_outputs)

        output = activations.reshape(original_shape)

        return output

    @torch.no_grad()
    def process_activations(self, activations: torch.Tensor) -> torch.Tensor:
        # activations: (batch_size, h*w, channels)
        encoded_activations = self.activation_encode(activations) \
            if self.activation_encode is not None \
            else activations

        encoded_activations[:, :, self.channels] = self.strength

        decoded_activations = self.activation_decode(encoded_activations) \
            if self.activation_decode is not None \
            else encoded_activations

        return decoded_activations



class SteeringHookFactory(BaseModel):
    """
    A Factory class for partial construction of SteeringHook instances.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sae: Optional[Sae] = None
    pipe_adapter: ModifiedDiffusionPipelineAdapter

    def create(self, *,
               channels: list[int],
               strength: torch.Tensor,
               timesteps: list[int]) -> SteeringHook:
        assert strength.shape[0] == len(channels), "Strength tensor shape must match number of channels."

        if self.sae is not None:
            def activation_encode(x: torch.Tensor):
                # TODO: move top k to decoder?
                # if shape mismatch happens in top k
                # original_shape = x.shape
                # x, _, _ = self.sae.preprocess_input(x)
                activations = self.sae.pre_acts(x)
                top_activations, top_indices = self.sae.select_topk(activations)

                # only preserving top k by setting the rest to zero
                preserved = torch.zeros_like(activations)
                preserved.scatter_(dim=-1, index=top_indices, src=top_activations)

                # preserved = preserved.reshape(original_shape)
                return preserved


            def activation_decode(x: torch.Tensor):
                return (x @ self.sae.W_dec) + self.sae.b_dec

            return SteeringHook(
                channels=channels,
                strength=strength,
                timesteps=timesteps,
                pipe_adapter=self.pipe_adapter,
                activation_encode=activation_encode,
                activation_decode=activation_decode
            )
        else:
            return SteeringHook(
                channels=channels,
                strength=strength,
                timesteps=timesteps,
                pipe_adapter=self.pipe_adapter
            )
