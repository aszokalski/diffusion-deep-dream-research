from diffusers import DiffusionPipeline
import torch

class ModifiedDiffusionPipelineAdapter:
    """
    An adapter for DiffusionPipeline. Exposes the modified pipeline as `pipe` property.
    - Overwrites the forward call of the UNet to set the current timestep.
    - Disables safety checker.
    """
    def __init__(self, pipe: DiffusionPipeline):
        original_forward = pipe.unet.forward

        def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
            t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            pipe.unet.current_timestep = t_val
            return original_forward(sample, timestep, encoder_hidden_states, **kwargs)

        pipe.unet.forward = intercepted_forward
        pipe.safety_checker = None

        self._pipe = pipe

    @property
    def pipe(self):
        return self._pipe