from typing import Optional, Any, Callable

import torch
from pydantic import PrivateAttr, BaseModel, ConfigDict
import torch.nn as nn

from diffusion_deep_dream_research.model.hooks.base_hook import BaseHook, EarlyExit
from diffusion_deep_dream_research.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter
from diffusion_deep_dream_research.utils.torch_utils import reshape_to_batch_spatial_channels

from submodules.SAeUron.SAE.sae import Sae


class CaptureHook(BaseHook):
    """An abstraction over a PyTorch hook that captures activations from a module.
    It's used to capture activations and expose them through the `get_last_activations()` method

    NOTE: This hook completely ignores the spatial dimensions of the activations.
    It computes the mean across all spatial dimensions and batches.
    """
    timesteps: list[int]
    detach: bool
    early_exit: bool
    pipe_adapter: ModifiedDiffusionPipelineAdapter
    activation_encode: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    _activations: dict[int, torch.Tensor] = PrivateAttr(default_factory=dict)

    def __call__(self, module: nn.Module, input: Any, output: Any):
        t = self.pipe_adapter.pipe.unet.current_timestep
        if t not in self.timesteps:
            return output

        if isinstance(output, tuple):
            loc_output = output[0]
        else:
            loc_output = output

        if self.detach:
            loc_output = loc_output.detach()

        loc_output = reshape_to_batch_spatial_channels(module, loc_output)

        self._activations[t] = self.process_activations(loc_output)

        if self.early_exit:
            raise EarlyExit()

        return output

    def process_activations(self, activations: torch.Tensor) -> torch.Tensor:
        # activations: (batch_size, h*w, channels)
        encoded_activations = self.activation_encode(
            activations) if self.activation_encode is not None else activations
        # The encoder (if specified) encodes the channel dimension of the activations
        return torch.mean(encoded_activations, dim=1)  # (batch_size, channels or encoded_channels,)

    def get_last_activations(self) -> dict[int, torch.Tensor]:
        """
        Returns the last activations dict captured by the hook.
        [timestep] -> activations (batch_size, channels,)
        """
        if self._activations is None:
            raise RuntimeError("No activations captured yet. Run the model first.")

        return self._activations

    def clear_activations(self):
        self._activations = {}

class CaptureHookFactory(BaseModel):
    """
    A Factory class for partial construction of CaptureHook instances.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sae: Optional[Sae] = None
    pipe_adapter: ModifiedDiffusionPipelineAdapter

    def create(self, *, timesteps: list[int], early_exit: bool, detach: bool) -> CaptureHook:
        if self.sae is not None:
            return CaptureHook(
                detach=detach,
                early_exit=early_exit,
                timesteps=timesteps,
                pipe_adapter=self.pipe_adapter,
                activation_encode=lambda x: self.sae.pre_acts(x),
            )
        else:
            return CaptureHook(
                detach=detach,
                early_exit=early_exit,
                timesteps=timesteps,
                pipe_adapter=self.pipe_adapter
            )
