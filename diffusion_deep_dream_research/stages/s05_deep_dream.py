import datetime
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]
from lightning import Fabric
from loguru import logger
from safetensors.torch import save_file
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from diffusion_deep_dream_research.config.config_schema import (
    ExperimentConfig,
    DeepDreamStageConfig, Timesteps
)
from diffusion_deep_dream_research.core.data.index_dataset import IndexDataset
from diffusion_deep_dream_research.core.model.hooked_model_wrapper import HookedModelWrapper
from diffusion_deep_dream_research.core.regularisation.gradient_transforms.gradient_preconditioning import \
    GradientPreconditioner
from diffusion_deep_dream_research.core.regularisation.gradient_transforms.gradient_smoothing import GradientSmoother
from diffusion_deep_dream_research.core.regularisation.latent_augumenter import LatentAugmenter
from diffusion_deep_dream_research.core.regularisation.penalties.base_penalty import BasePenalty
from diffusion_deep_dream_research.core.regularisation.penalties.moment_penalty import MomentPenalty
from diffusion_deep_dream_research.core.regularisation.penalties.range_penalty import RangePenalty
from diffusion_deep_dream_research.core.regularisation.penalties.total_variation_penalty import TotalVariationPenalty
from diffusion_deep_dream_research.utils.config_utils import resolve_sae_config
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging
from diffusion_deep_dream_research.utils.prior_results_reading_utils import get_prior_results
from diffusion_deep_dream_research.utils.torch_utils import get_dtype, generate_random_priors_from_seeds
from submodules.SAeUron.SAE.sae import Sae


def generate_deep_dreams_for_channel_timestep(
        *,
        stage_config: DeepDreamStageConfig,
        model_wrapper: HookedModelWrapper,
        channel: int,
        timestep: int,
        priors: torch.Tensor
) -> torch.Tensor:
    priors.requires_grad_(True)
    latents = priors # Initial latents
    optimizer = torch.optim.AdamW([latents], lr=stage_config.learning_rate)
    scaler = GradScaler(enabled=(priors.device.type == "cuda"))

    augumenter = LatentAugmenter(
        jitter_max=stage_config.jitter_max,
        rotate_max=stage_config.rotate_max,
        scale_max=stage_config.scale_max
    )

    gradient_preconditioner = GradientPreconditioner(
        latent_height=priors.shape[-2],
        latent_width=priors.shape[-1],
        device=priors.device
    )

    gradient_smoother = GradientSmoother(
        kernel_size=stage_config.gradient_smoothing_kernel_size,
        sigma_start=stage_config.gradient_smoothing_sigma_start,
        sigma_end=stage_config.gradient_smoothing_sigma_end,
        num_steps=stage_config.num_steps
    )

    penalties: list[BasePenalty] = [
        TotalVariationPenalty(weight=stage_config.total_variation_penalty_weight),
        RangePenalty(weight=stage_config.range_penalty_weight, threshold=stage_config.range_penalty_threshold),
        MomentPenalty(weight=stage_config.moment_penalty_weight),
    ]



    for step in range(stage_config.num_steps):
        optimizer.zero_grad()

        z = augumenter(latents)

        if stage_config.see_through_schedule_noise:
            z = model_wrapper.apply_scheduler_noise(z, timestep=timestep)

        with autocast(enabled=(priors.device.type == "cuda"), device_type=priors.device.type):
            # Forward pass
            activation = model_wrapper.activation(
                z=z,
                channel=channel,
                timestep=timestep,
            )

            loss = -activation
            for penalty in penalties:
                penalty_value = penalty(priors)
                loss = loss + penalty_value

            #Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if stage_config.use_decorrelated_space:
                latents.grad = gradient_preconditioner(latents.grad)

            gradient_smoother.update_current_step(step)
            latents.grad = gradient_smoother(latents.grad)

            scaler.step(optimizer)
            scaler.update()

    return latents

def generate_deep_dreams(
        *,
        config: ExperimentConfig,
        stage_config: DeepDreamStageConfig,
        timesteps_analysis_results_abs_path: Path,
        prior_results_abs_path: Path,
        sae: bool,
        fabric: Fabric
):
    # Model setup
    dtype = get_dtype()

    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_to_analyse.path,
        torch_dtype=dtype,
    )
    pipe.to(fabric.device)

    # Disable the progress bar for non-local runs
    pipe.set_progress_bar_config(disable=config.infrastructure_name != "local")

    if sae:
        # NOTE: SAE config is very SAeUron specific at the moment. Could be made more general
        sae_path = f"{config.sae.path}/unet.{config.target_layer_name}"
        sae_model = Sae.load_from_disk(
            path=sae_path,
            device=fabric.device,
            decoder=True
        )

        model_wrapper = HookedModelWrapper.from_sae(
            pipe=pipe,
            target_layer_name=config.target_layer_name,
            sae=sae_model
        )
    else:
        model_wrapper = HookedModelWrapper.from_layer(
            pipe=pipe,
            target_layer_name=config.target_layer_name,
        )

    # Data and parameters setup
    suffix = "_sae" if sae else ""
    active_timesteps_json_filename = f"active_timesteps{suffix}.json"

    logger.info(f"Loading analysis data from {timesteps_analysis_results_abs_path}...")

    # Load Timesteps Analysis Results
    with open(timesteps_analysis_results_abs_path / active_timesteps_json_filename, "r") as f:
        active_timesteps = json.load(f)  # channel -> list of timesteps

    use_active_timesteps = Timesteps.active_timesteps in stage_config.timesteps
    additional_timesteps = [
        t for t in stage_config.timesteps
        if isinstance(t, int)
    ]

    n_channels = len(active_timesteps)

    # Load Prior Results
    prior_results = get_prior_results(prior_results_abs_path)
    curr_prior_results = prior_results.sae if sae else prior_results.raw

    start_channel = stage_config.start_channel if stage_config.start_channel is not None else 0
    end_channel = stage_config.end_channel if stage_config.end_channel is not None else n_channels - 1
    logger.info(
        f"Generating priors for channels {start_channel} to {end_channel} using timesteps: {stage_config.timesteps}")

    # This is a workaround which allows to run this distributed across multiple GPUs

    channels_dataset = IndexDataset(start_channel, end_channel + 1)
    data_loader = DataLoader(channels_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    data_loader = fabric.setup_dataloaders(data_loader)

    rank_dir = Path(f"fabric_rank_{fabric.global_rank}")
    rank_dir.mkdir(parents=True, exist_ok=True)

    output_dir = rank_dir / f"priors{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving priors to {output_dir}")

    model_wrapper.eval()

    start_time = time.time()
    with torch.no_grad():
        for i, single_channel_batch in enumerate(data_loader):
            channel = single_channel_batch[0]
            channel_name = f"channel_{channel:04d}"
            channel_path = output_dir / channel_name
            channel_path.mkdir(parents=True, exist_ok=True)

            channel_done_marker = channel_path / ".done"
            if channel_done_marker.exists():
                logger.info(f"Rank {fabric.global_rank}: Channel {channel} already processed. Skipping.")
                continue

            if stage_config.use_prior:
                priors = curr_prior_results[channel].get_latents(
                    device=fabric.device,
                    dtype=dtype
                )
            else:
                priors = generate_random_priors_from_seeds(
                    seeds=stage_config.seeds,
                    in_channels=pipe.unet.config.in_channels,
                    sample_size=pipe.unet.sample_size,
                    device=fabric.device,
                    dtype=dtype
                )

            timesteps = set(active_timesteps[channel]) | set(additional_timesteps)

            for timestep in timesteps:
                timestep_path = channel_path / f"timestep_{timestep:04d}"
                timestep_done_marker = timestep_path / ".done"
                if timestep_done_marker.exists():
                    logger.info(
                        f"Rank {fabric.global_rank}: Channel {channel}, Timestep {timestep} already processed. Skipping.")
                    continue

                latents = generate_deep_dreams_for_channel_timestep(
                    stage_config=stage_config,
                    model_wrapper=model_wrapper,
                    channel=channel,
                    timestep=timestep,
                    priors=priors
                )

                latents = latents.detach()
                images = model_wrapper.decode_latents(latents)
                latents = latents.cpu()


                latents_dir = timestep_path / "latents"
                latents_dir.mkdir(parents=True, exist_ok=True)

                images_dir = timestep_path / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                for j, (image, latent) in enumerate(zip(images, latents)):
                    image = (image * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image)
                    image_pil.save(images_dir / f"deep_dream_image_{j:04d}.png")

                    save_file(
                        {"latent": latent},
                        latents_dir / f"deep_dream_latent_{j:04d}.safetensors"
                    )

                timestep_done_marker.touch()

            if i % stage_config.log_every_n_steps == 0:
                current_time = time.time()
                elapsed_seconds = current_time - start_time
                batches_processed = i + 1

                avg_seconds_per_batch = elapsed_seconds / batches_processed if batches_processed > 0 else 0

                remaining_batches = len(channels_dataset) - batches_processed
                eta_seconds = remaining_batches * avg_seconds_per_batch
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    f"Rank {fabric.global_rank}: Processed {batches_processed}/{len(channels_dataset)} channels. ETA: {eta_str}")

            channel_done_marker.touch()

    logger.info(f"Rank {fabric.global_rank}: Done! (sae: {sae})")
    fabric.barrier()



def run_deep_dream(config: ExperimentConfig):
    logger.info("Starting Fabric...")
    fabric = Fabric(
        accelerator=config.fabric.accelerator,
        devices="auto",
        strategy="ddp"
    )
    fabric.launch()

    setup_distributed_logging(fabric.global_rank)

    stage_config = cast(DeepDreamStageConfig, config.stage_config)
    sae_stage_config = resolve_sae_config(stage_config)
    use_sae = config.use_sae


    if not stage_config.use_prior:
        if stage_config.seeds is None or stage_config.n_results is None:
            raise ValueError("seeds and n_results must be provided in stage_config when use_prior is False")
        elif len(stage_config.seeds) != stage_config.n_results:
            raise ValueError("Length of seeds and n_results must be the same")


    timesteps_analysis_results_abs_path = config.outputs_dir / stage_config.timestep_analysis_results_dir
    prior_results_abs_path = config.outputs_dir / stage_config.prior_results_dir

    generate_deep_dreams(
        config=config,
        stage_config=stage_config,
        timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
        prior_results_abs_path=prior_results_abs_path,
        sae=False,
        fabric=fabric
    )

    if use_sae:
        generate_deep_dreams(
            config=config,
            stage_config=sae_stage_config,
            timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
            prior_results_abs_path=prior_results_abs_path,
            sae=True,
            fabric=fabric
        )

    logger.info("Done!")

