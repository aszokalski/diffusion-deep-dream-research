from functools import lru_cache
from typing import Callable, ContextManager, Optional, Literal, NamedTuple

import torch
from PIL.Image import Image
from diffusers import DiffusionPipeline

from diffusion_deep_dream_research.core.hooks.base_hook import BaseHook, create_target_hook_context, EarlyExit
from diffusion_deep_dream_research.core.hooks.capture_hook import CaptureHookFactory, CaptureHook
from diffusion_deep_dream_research.core.hooks.steering_hook import SteeringHookFactory
from diffusion_deep_dream_research.core.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter
import torch.nn as nn

from submodules.SAeUron.SAE.sae import Sae


class HookedModelWrapper(nn.Module):
    def __init__(self,
                 *,
                 pipe_adapter: ModifiedDiffusionPipelineAdapter,
                 target_hook_context: Callable[[BaseHook], ContextManager[BaseHook]],
                 capture_hook_factory: CaptureHookFactory,
                 steering_hook_factory: SteeringHookFactory
                 ):
        super().__init__()
        self.pipe_adapter = pipe_adapter
        self.capture_hook_factory = capture_hook_factory
        self.steering_hook_factory = steering_hook_factory
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

        steering_hook_factory = SteeringHookFactory(
            pipe_adapter=pipe_adapter
        )

        return cls(
            pipe_adapter=pipe_adapter,
            target_hook_context=target_hook_context,
            capture_hook_factory=capture_hook_factory,
            steering_hook_factory=steering_hook_factory
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
        steering_hook_factory = SteeringHookFactory(
            pipe_adapter=pipe_adapter,
            sae=sae
        )
        return cls(
            pipe_adapter=pipe_adapter,
            target_hook_context=target_hook_context,
            capture_hook_factory=capture_hook_factory,
            steering_hook_factory=steering_hook_factory
        )

    def forward(self,
                *,
                prompts: list[str],
                num_images_per_prompt: int = 1,
                seeds: Optional[list[int]] = None,
                output_type: Optional[Literal["pil", "latent"]] = "pil",
                **kwargs) -> torch.Tensor:
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
        images: list[Image]
        hook_activations: dict[int, dict[CaptureHook.ActivationType, torch.Tensor]]

    @torch.no_grad()
    def forward_with_capture(self,
                             *,
                             prompts: list[str],
                             num_images_per_prompt: int = 1,
                             ) -> ForwardWithCaptureResult:
        """
        Performs a forward pass for n prompts and m num_images_per_prompt and returns:
        - images [batch_size * num_images_per_prompt]
        - hook_activations: dict[timestep] -> activations (batch_size * num_images_per_prompt [total_batch_size], channels,)
        """
        with self.target_hook_context(
                self.capture_hook_factory.create(
                    timesteps=self.pipe_adapter.pipe.scheduler.timesteps.tolist(),
                    detach=True,
                    early_exit=False
        )) as hook:
            images = self(prompts=prompts, num_images_per_prompt=num_images_per_prompt, output_type="pil")

        hook_activations = hook.get_last_activations()
        return HookedModelWrapper.ForwardWithCaptureResult(images, hook_activations)

    @torch.no_grad()
    def steer(self,
              *,
              channel: int,
              strength: float,
              timesteps: list[int],
              n_results: int = 1,
              seeds: Optional[list[int]] = None,
              output_type: Optional[Literal["pil", "latent"]] = "pil"
              ) -> torch.Tensor:
        """
        Generates steered images by exciting `channel` with `strength`
        at `timesteps` with an empty prompt.

        :param channel: Channel to steer.
        :param strength: Strength of steering. The activations of the selected `channel` across all batches and spatial dimentions are set to `strength`.
        :param timesteps: Timesteps to steer at. Should be the ones that were the most active for this channel during a dataset run.
        :param n_results: Number of results to generate.
        :param seeds: Seed for each result
        :param output_type: Latent or PIL image.
        :return: Steered images or latents [n_results]
        """

        with self.target_hook_context(
                self.steering_hook_factory.create(
                    channel=channel,
                    strength=strength,
                    timesteps=timesteps,
        )):
            return self(
                prompts=[""]*n_results,
                seeds=seeds,
                output_type=output_type,
                guidance_scale=1.0
            )


    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        A function to encode images into latent space using the VAE encoder.
        :param images:
        :return:
        """
        latent_dist = self.pipe_adapter.pipe.vae.encode(images).latent_dist
        latents = latent_dist.sample()
        return latents * self.pipe_adapter.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        A function to decode latent representations back into images using the VAE decoder.
        :param latents:
        :return:
        """
        latents = latents / self.pipe_adapter.pipe.vae.config.scaling_factor
        images = self.pipe_adapter.pipe.vae.decode(latents.to(self.pipe_adapter.pipe.vae.dtype)).sample
        cpu_images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        #normalizing images to [0, 1]
        normalized_images = (cpu_images - cpu_images.min()) / (cpu_images.max() - cpu_images.min())
        return normalized_images

    @lru_cache
    @torch.no_grad()
    def _empty_embeddings(self, batch_size: int) -> torch.Tensor:
        """
        Helper function to return empty embeddings (equivalent to empty prompt)
        It's cached so we don't have to recompute it every time.

        It's done this way to support multiple types of encoders.

        :param batch_size: The batch size to return embeddings for.
        :return: Empty embeddings tensor of shape (batch_size, hidden_dim)
        """
        _, negative_embeds = self.pipe_adapter.pipe.encode_prompt(
            prompt="",
            device=self.device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            negative_prompt=None
        )
        return negative_embeds

    def activation(self,
                   *,
                   z: torch.Tensor,
                   channel: int,
                   timestep: int
                   ) -> torch.Tensor:
        """
        Activation function for a given timestep and channel used for deep dreams.
        Returns the mean activation for the selected channel.
        :param z: Latent input tensor of shape (batch_size, channels, height, width)
        :param channel: Channel to extract activation from.
        :param timestep: Timestep to extract activation from.
        :return: Activation tensor of shape (1,)
        """
        with self.target_hook_context(
                self.capture_hook_factory.create(
                    timesteps=[timestep],
                    detach=False,
                    early_exit=True
        )) as hook:
            batch_size = z.shape[0]
            embeddings = self._empty_embeddings(batch_size)

            try:
                self.pipe_adapter.pipe.unet(
                    z,
                    timestep=timestep,
                    encoder_hidden_states=embeddings
                )
            except EarlyExit:
                pass

            # mean over batch for the selected channel
            return torch.mean(hook.get_last_activations()[timestep][:, channel])

    def apply_scheduler_noise(self, z: torch.Tensor, *, timestep: int) -> torch.Tensor:
        noise = torch.randn_like(z)
        noisy_z = self.pipe_adapter.pipe.scheduler.add_noise(
            original_samples=z,
            noise=noise,
            timesteps=torch.tensor([timestep], device=self.device).int()
        )
        return noisy_z