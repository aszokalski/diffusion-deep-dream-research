from typing import Callable

from diffusers import DiffusionPipeline
import torch


class ModifiedDiffusionPipelineAdapter:
    """
    An adapter for DiffusionPipeline. Exposes the modified pipeline as `pipe` property.
    - Overwrites the forward call of the UNet to set the current timestep.
    - Disables safety checker.
    - Sets the number of timesteps to 50.
    """

    def __init__(self, pipe: DiffusionPipeline):
        # Assertions to please static type checkers reasonably where possible
        assert hasattr(pipe, "unet"), "The pipeline must have a UNet model."
        assert isinstance(pipe.unet, torch.nn.Module), (
            "The UNet must be an instance of torch.nn.Module."
        )
        assert hasattr(pipe, "scheduler"), "The pipeline must have a scheduler."
        assert hasattr(pipe.scheduler, "set_timesteps") and callable(
            pipe.scheduler.set_timesteps
        ), "The scheduler must have a callable set_timesteps method."

        original_forward: Callable = pipe.unet.forward

        pipe.scheduler.set_timesteps(50)

        def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
            t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            pipe.unet.current_timestep = t_val  # ty:ignore[unresolved-attribute]
            return original_forward(sample, timestep, encoder_hidden_states, **kwargs)

        pipe.unet.forward = intercepted_forward  # ty:ignore[invalid-assignment]
        pipe.safety_checker = None  # ty:ignore[invalid-assignment]

        self._pipe = pipe

    @property
    def pipe(self):
        return self._pipe
