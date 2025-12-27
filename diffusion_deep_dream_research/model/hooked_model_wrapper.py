from typing import Callable, ContextManager, Optional, Literal, NamedTuple

import torch
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
        self.device = pipe_adapter.pipe.device
        self.dtype = pipe_adapter.pipe.unet.dtype

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

    def forward(self,
                *,
                prompts: list[str],
                num_images_per_prompt: int = 1,
                seeds: Optional[list[int]] = None,
                output_type: Optional[Literal["pil", "latent"]] = "pil",
                **kwargs):
        batch_size = len(prompts)
        total_images = batch_size * num_images_per_prompt
        generators = None

        if seeds is not None:
            if len(seeds) == num_images_per_prompt:
                # list of seeds to be reused for each prompt
                expanded_seeds = seeds * batch_size

            elif len(seeds) == total_images:
                # each image for each prompt has its own seed
                expanded_seeds = seeds

            else:
                raise ValueError(
                    f"Seed mismatch! You requested {total_images} total images "
                    f"({batch_size} prompts * {num_images_per_prompt} img/prompt), "
                    f"but provided {len(seeds)} seeds. "
                    f"Please provide either {num_images_per_prompt} (to reuse) "
                    f"or {total_images} (explicit) seeds."
                )

            generators = [
                torch.Generator(device=self.device).manual_seed(s)
                for s in expanded_seeds
            ]

        output = self.pipe_adapter.pipe(
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            generator=generators,
            output_type=output_type,
            **kwargs
        )

        return output.images

    class ForwardWithCaptureResult(NamedTuple):
        images: torch.Tensor
        hook_activations: dict[int, torch.Tensor]

    def forward_with_capture(self,
                             *,
                             prompts: list[str],
                             num_images_per_prompt: int = 1,
                             ) -> ForwardWithCaptureResult:
        """
        Performs a forward pass for n prompts and m num_images_per_prompt and returns:
        - images [n*m]
        - hook_activations: dict[timestep] -> activations (batch_size [n*m], channels,)
        """
        with self.target_hook_context(self.capture_hook_factory.create(
            timesteps=self.pipe_adapter.pipe.scheduler.timesteps.tolist(),
            detach=True,
            early_exit=False
        )) as hook:
            images = self(prompts=prompts, num_images_per_prompt=num_images_per_prompt)

        hook_activations = hook.get_last_activations()
        return HookedModelWrapper.ForwardWithCaptureResult(images, hook_activations)




