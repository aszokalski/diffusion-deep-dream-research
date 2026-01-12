import datetime
import json
import pickle
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
from torch.utils.data import DataLoader

from diffusion_deep_dream_research.config.config_schema import (
    ExperimentConfig,
    Timesteps,
    PriorStageConfig,
)
from diffusion_deep_dream_research.core.data.index_dataset import IndexDataset
from diffusion_deep_dream_research.core.model.hooked_model_wrapper import HookedModelWrapper
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging
from diffusion_deep_dream_research.utils.torch_utils import get_dtype
from submodules.SAeUron.SAE.sae import Sae


def generate_priors(
        *,
        config: ExperimentConfig,
        stage_config: PriorStageConfig,
        timesteps_analysis_results_abs_path: Path,
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
        #NOTE: SAE config is very SAeUron specific at the moment. Could be made more general
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
    pkl_filename = f"frequency_in_top_k_sorted_timesteps_max_activation{suffix}.pkl"
    active_timesteps_json_filename = f"active_timesteps{suffix}.json"

    logger.info(f"Loading analysis data from {timesteps_analysis_results_abs_path}...")

    # Load Data
    with open(timesteps_analysis_results_abs_path / pkl_filename, "rb") as f:
        _, sorted_timesteps, max_activation = pickle.load(f) # Could be done more efficiently

    with open(timesteps_analysis_results_abs_path / active_timesteps_json_filename, "r") as f:
       active_timesteps = json.load(f) # channel -> list of timesteps

    n_channels = max_activation.shape[0]
    start_channel = stage_config.start_channel if stage_config.start_channel is not None else 0
    end_channel = stage_config.end_channel if stage_config.end_channel is not None else n_channels - 1
    logger.info(f"Generating priors for channels {start_channel} to {end_channel} using timesteps: {stage_config.timesteps}")

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

            done_marker = channel_path / ".done"
            if done_marker.exists():
                logger.info(f"Rank {fabric.global_rank}: Channel {channel} already processed. Skipping.")
                continue

            if stage_config.timesteps == Timesteps.active_timesteps:
                steer_timesteps = active_timesteps[channel]
            elif stage_config.timesteps == Timesteps.all_timesteps:
                steer_timesteps = sorted_timesteps
            

            if sae and stage_config.steer_strength_scale_sae is not None:
                steer_strength = stage_config.steer_strength_scale_sae
            else:
                steer_strength = stage_config.steer_strength_scale

            latents = model_wrapper.steer(
                channel = channel,
                strength={
                    ts: max_act * steer_strength
                    for ts, max_act in zip(sorted_timesteps, max_activation[channel])
                },
                timesteps=steer_timesteps,
                n_results=stage_config.n_results,
                seeds=stage_config.seeds,
                output_type="latent"
            )

            latents = latents.detach()
            images = model_wrapper.decode_latents(latents)
            latents = latents.cpu() # Moving to cpu to save on disk

            latents_dir = channel_path / "latents"
            latents_dir.mkdir(parents=True, exist_ok=True)

            images_dir = channel_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            for j, (image, latent) in enumerate(zip(images, latents)):
                image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
                image_pil.save(images_dir / f"prior_image_{j:04d}.png")

                save_file(
                    {"latent": latent},
                    latents_dir / f"prior_latent_{j:04d}.safetensors"
                )

            if i % stage_config.log_every_n_steps == 0:
                current_time = time.time()
                elapsed_seconds = current_time - start_time
                batches_processed = i + 1

                avg_seconds_per_batch = elapsed_seconds / batches_processed if batches_processed > 0 else 0

                remaining_batches = len(channels_dataset) - batches_processed
                eta_seconds = remaining_batches * avg_seconds_per_batch
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(f"Rank {fabric.global_rank}: Processed {batches_processed}/{len(channels_dataset)} channels. ETA: {eta_str}")

            done_marker.touch()

    logger.info(f"Rank {fabric.global_rank}: Done! (sae: {sae})")
    fabric.barrier()


def run_prior(config: ExperimentConfig):
    logger.info("Starting Fabric...")
    fabric = Fabric(
        accelerator=config.fabric.accelerator,
        devices="auto",
        strategy="ddp"
    )
    fabric.launch()

    setup_distributed_logging(fabric.global_rank)

    stage_config = cast(PriorStageConfig, config.stage_config)
    use_sae = config.use_sae

    timesteps_analysis_results_abs_path = config.outputs_dir / stage_config.timestep_analysis_results_dir

    generate_priors(
        config=config,
        stage_config=stage_config,
        timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
        sae=False,
        fabric=fabric
    )

    if use_sae:
        generate_priors(
            config=config,
            stage_config=stage_config,
            timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
            sae=True,
            fabric=fabric
        )

    logger.info("Done!")

