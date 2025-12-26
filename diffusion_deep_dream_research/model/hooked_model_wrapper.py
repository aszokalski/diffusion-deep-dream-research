from typing import Callable, ContextManager

from diffusers import DiffusionPipeline

from diffusion_deep_dream_research.model.hooks.base_hook import BaseHook, create_target_hook_context
from diffusion_deep_dream_research.model.hooks.capture_hook import CaptureHookFactory
from diffusion_deep_dream_research.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter
import torch.nn as nn

from submodules.SAeUron.SAE.sae import Sae


class HookedModelWrapper(nn.Module):
    def __init__(self,
                 *,
                 pipe_adapter: ModifiedDiffusionPipelineAdapter,
                 target_hook_context: Callable[[BaseHook], ContextManager[BaseHook]],
                 capture_hook_factory: CaptureHookFactory
                 ):
        super().__init__()
        self.pipe_adapter = pipe_adapter
        self.capture_hook_factory = capture_hook_factory
        self.target_hook_context = target_hook_context

    @classmethod
    def from_layer(cls,
                   *,
                   pipe: DiffusionPipeline,
                   target_layer_name: str
                   ):
        pipe_adapter = ModifiedDiffusionPipelineAdapter(pipe)

        target_layer = dict(pipe.unet.named_modules()).get(target_layer_name)
        assert target_layer is not None, f"Layer {target_layer_name} not found in model."

        target_hook_context = create_target_hook_context(target_layer)

        capture_hook_factory = CaptureHookFactory(
            pipe_adapter=pipe_adapter
        )
        return cls(
            pipe_adapter=pipe_adapter,
            target_hook_context=target_hook_context,
            capture_hook_factory=capture_hook_factory
        )

    @classmethod
    def from_sae(cls,
                 *,
                 pipe: DiffusionPipeline,
                 target_layer_name: str,
                 sae: Sae
                 ):
        pipe_adapter = ModifiedDiffusionPipelineAdapter(pipe)

        target_layer = dict(pipe.unet.named_modules()).get(target_layer_name)
        assert target_layer is not None, f"Layer {target_layer_name} not found in model."

        target_hook_context = create_target_hook_context(target_layer)

        capture_hook_factory = CaptureHookFactory(
            pipe_adapter=pipe_adapter,
            sae=sae
        )
        return cls(
            pipe_adapter=pipe_adapter,
            target_hook_context=target_hook_context,
            capture_hook_factory=capture_hook_factory
        )

    