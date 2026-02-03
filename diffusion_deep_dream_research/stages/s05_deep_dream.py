import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, cast

from diffusers import StableDiffusionPipeline
from lightning import Fabric
from loguru import logger
import numpy as np
from PIL import Image
from safetensors.torch import save_file
from submodules.SAeUron.SAE.sae import Sae
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from diffusion_deep_dream_research.config.config_schema import (
    DeepDreamStageConfig,
    ExperimentConfig,
    Timesteps,
)
from diffusion_deep_dream_research.core.data.index_dataset import IndexDataset
from diffusion_deep_dream_research.core.hooks.capture_hook import CaptureHook
from diffusion_deep_dream_research.core.model.hooked_model_wrapper import HookedModelWrapper
from diffusion_deep_dream_research.core.regularisation.gradient_transforms.gradient_preconditioning import (
    GradientPreconditioner,
)
from diffusion_deep_dream_research.core.regularisation.gradient_transforms.gradient_smoothing import (
    GradientSmoother,
)
from diffusion_deep_dream_research.core.regularisation.latent_augumenter import LatentAugmenter
from diffusion_deep_dream_research.core.regularisation.penalties.base_penalty import BasePenalty
from diffusion_deep_dream_research.core.regularisation.penalties.moment_penalty import (
    MomentPenalty,
)
from diffusion_deep_dream_research.core.regularisation.penalties.range_penalty import RangePenalty
from diffusion_deep_dream_research.core.regularisation.penalties.total_variation_penalty import (
    TotalVariationPenalty,
)
from diffusion_deep_dream_research.utils.config_utils import resolve_sae_config
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging
from diffusion_deep_dream_research.utils.prior_results_reading_utils import get_prior_results
from diffusion_deep_dream_research.utils.torch_utils import (
    generate_random_priors_from_seeds,
    get_dtype,
)


def generate_deep_dreams_for_channel_timestep(
    *,
    stage_config: DeepDreamStageConfig,
    model_wrapper: HookedModelWrapper,
    channel: int,
    timestep: int,
    activation_type: CaptureHook.ActivationType,
    priors: torch.Tensor,
    output_dir: Path,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    latents = priors.detach().clone().float()
    latents.requires_grad_(True)

    optimizer = torch.optim.AdamW([latents], lr=stage_config.learning_rate)
    scaler = GradScaler(enabled=(priors.device.type == "cuda"))

    augumenter = LatentAugmenter(
        jitter_max=stage_config.jitter_max,
        rotate_max=stage_config.rotate_max,
        scale_max=stage_config.scale_max,
    )

    gradient_preconditioner = GradientPreconditioner(
        latent_height=priors.shape[-2], latent_width=priors.shape[-1], device=priors.device
    )

    gradient_smoother = GradientSmoother(
        kernel_size=stage_config.gradient_smoothing_kernel_size,
        sigma_start=stage_config.gradient_smoothing_sigma_start,
        sigma_end=stage_config.gradient_smoothing_sigma_end,
        num_steps=stage_config.num_steps,
    )

    penalties: list[BasePenalty] = [
        TotalVariationPenalty(weight=stage_config.total_variation_penalty_weight),
        RangePenalty(
            weight=stage_config.range_penalty_weight,
            threshold=stage_config.range_penalty_threshold,
        ),
        MomentPenalty(weight=stage_config.moment_penalty_weight),
    ]

    logger.debug(f"Starting optimization loop for {stage_config.num_steps} steps.")

    for step in range(stage_config.num_steps):
        optimizer.zero_grad()

        z = augumenter(latents)

        if stage_config.see_through_schedule_noise:
            z = model_wrapper.apply_scheduler_noise(z, timestep=timestep)

        with autocast(enabled=(priors.device.type == "cuda"), device_type=priors.device.type):
            # Forward pass
            activation = model_wrapper.activation(
                z=z,
                activation_type=activation_type,
                channel=channel,
                timestep=timestep,
            )

            loss = -activation

            step_stats: Dict[str, Any] = {
                "step": step,
                "activation": activation.item(),
                "penalties": {},
            }

            for penalty in penalties:
                penalty_value = penalty(latents)
                loss = loss + penalty_value
                step_stats["penalties"][penalty.__class__.__name__] = penalty_value.item()

            step_stats["total_loss"] = loss.item()

            should_save_intermediate = (
                stage_config.intermediate_opt_results_every_n_steps > 0
                and step % stage_config.intermediate_opt_results_every_n_steps == 0
            )

            if should_save_intermediate:
                step_dir = output_dir / "intermediate" / f"step_{step:04d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Saving intermediate result at step {step} to {step_dir}")

                with open(step_dir / "stats.json", "w") as f:
                    json.dump(step_stats, f, indent=4)

                with torch.no_grad():
                    current_images = model_wrapper.decode_latents(latents.detach())
                    for j, image_arr in enumerate(current_images):
                        image_u8 = (image_arr * 255).astype(np.uint8)
                        Image.fromarray(image_u8).save(step_dir / f"image_{j:04d}.png")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            assert latents.grad is not None, "Latents gradient is None after backward pass."
            if stage_config.use_gradient_spectral_filtering:
                latents.grad = gradient_preconditioner(latents.grad)

            gradient_smoother.update_current_step(step)
            latents.grad = gradient_smoother(latents.grad)

            scaler.step(optimizer)
            scaler.update()

    logger.info(f"Done generating deep dream for channel {channel}, timestep {timestep}.")

    return latents, step_stats


def generate_deep_dreams(
    *,
    config: ExperimentConfig,
    stage_config: DeepDreamStageConfig,
    timesteps_analysis_results_abs_path: Path,
    prior_results_abs_path: Path,
    sae: bool,
    fabric: Fabric,
):
    logger.info(f"Starting generate_deep_dreams (SAE={sae})...")

    # Model setup
    dtype = get_dtype()
    logger.debug(f"Using dtype: {dtype}")

    logger.info(f"Loading Stable Diffusion model from {config.model_to_analyse.path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_to_analyse.path,
        torch_dtype=dtype,
    )
    pipe.to(fabric.device)

    # Disable the progress bar for non-local runs
    pipe.set_progress_bar_config(disable=config.infrastructure_name != "local")

    if sae:
        sae_path = f"{config.sae.path}/unet.{config.target_layer_name}"
        logger.info(f"Loading SAE model from {sae_path}...")
        sae_model = Sae.load_from_disk(path=sae_path, device=fabric.device, decoder=True)

        model_wrapper = HookedModelWrapper.from_sae(
            pipe=pipe, target_layer_name=config.target_layer_name, sae=sae_model
        )
        activation_type = CaptureHook.ActivationType.ENCODED
    else:
        logger.info(f"Wrapping model layer: {config.target_layer_name}")
        model_wrapper = HookedModelWrapper.from_layer(
            pipe=pipe,
            target_layer_name=config.target_layer_name,
        )
        activation_type = CaptureHook.ActivationType.RAW

    # Data and parameters setup
    suffix = "_sae" if sae else ""
    active_timesteps_json_filename = f"active_timesteps{suffix}.json"
    activity_peaks_json_filename = f"activity_peaks{suffix}.json"

    logger.info(f"Loading analysis data from {timesteps_analysis_results_abs_path}...")

    # Load Timesteps Analysis Results
    try:
        with open(timesteps_analysis_results_abs_path / active_timesteps_json_filename, "r") as f:
            active_timesteps = json.load(f)
        logger.debug(
            f"Loaded {len(active_timesteps)} entries from {active_timesteps_json_filename}"
        )

        with open(timesteps_analysis_results_abs_path / activity_peaks_json_filename, "r") as f:
            activity_peaks = json.load(f)
        logger.debug(f"Loaded {len(activity_peaks)} entries from {activity_peaks_json_filename}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load timestep analysis files: {e}")
        raise

    extend_timesteps_map = None
    if Timesteps.activity_peaks in stage_config.timesteps:
        logger.info("Using activity peaks for timesteps.")
        extend_timesteps_map = activity_peaks
    elif Timesteps.active_timesteps in stage_config.timesteps:
        logger.info("Using active timesteps for timesteps.")
        extend_timesteps_map = active_timesteps

    additional_timesteps = [t for t in stage_config.timesteps if isinstance(t, int)]
    logger.debug(f"Fixed additional timesteps: {additional_timesteps}")

    n_channels = len(active_timesteps)
    logger.info(f"Total number of channels detected: {n_channels}")

    # Load Prior Results
    logger.info(f"Loading prior results from {prior_results_abs_path}...")
    prior_results = get_prior_results(prior_results_abs_path)
    curr_prior_results = prior_results.sae if sae else prior_results.raw
    logger.info(f"Loaded priors for {len(curr_prior_results)} channels.")

    start_channel = stage_config.start_channel if stage_config.start_channel is not None else 0
    end_channel = (
        stage_config.end_channel if stage_config.end_channel is not None else n_channels - 1
    )
    logger.info(
        f"Generating deep_dream for channels {start_channel} to {end_channel} using timesteps config: {stage_config.timesteps}"
    )

    # Config for small scale sweeps.
    # It was added during experiments for convenience.
    if not sae and stage_config.channels is not None:
        logger.info(f"Using explicit channel list: {stage_config.channels}")
        channels_dataset = IndexDataset(stage_config.channels)
    elif sae and stage_config.channels_sae is not None:
        logger.info(f"Using explicit SAE channel list: {stage_config.channels_sae}")
        channels_dataset = IndexDataset(stage_config.channels_sae)
    else:
        logger.info(f"Using channel range [{start_channel}, {end_channel}]")
        channels_dataset = IndexDataset(start_channel, end_channel + 1)

    data_loader = DataLoader(channels_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    data_loader = fabric.setup_dataloaders(data_loader)

    rank_dir = Path(f"fabric_rank_{fabric.global_rank}")
    rank_dir.mkdir(parents=True, exist_ok=True)

    output_dir = rank_dir / f"deep_dream{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving deep_dream results to {output_dir}")

    model_wrapper.eval()

    start_time = time.time()

    for i, single_channel_batch in enumerate(data_loader):
        channel = single_channel_batch[0]
        logger.debug(f"Processing channel {channel}...")

        if stage_config.use_prior:
            if channel not in curr_prior_results:
                logger.warning(f"No priors found for channel {channel} (SAE={sae}). Skipping.")
                continue

            priors = curr_prior_results[channel].get_latents(
                device=fabric.device, dtype=dtype, n_results=stage_config.n_results
            )
            logger.debug(f"Loaded priors for channel {channel}: shape {priors.shape}")
        else:
            assert stage_config.seeds is not None, "Seeds must be provided when not using priors."
            logger.debug(f"Generating random priors from seeds: {stage_config.seeds}")

            priors = generate_random_priors_from_seeds(
                seeds=stage_config.seeds,
                in_channels=pipe.unet.config.in_channels,
                sample_size=pipe.unet.sample_size,
                device=fabric.device,
                dtype=dtype,
            )

        timesteps = set(additional_timesteps)
        if extend_timesteps_map is not None:
            timesteps.update(extend_timesteps_map[channel])

        if stage_config.use_just_one_timestep:
            timesteps = set(list(timesteps)[:1])
            logger.debug("Limiting to 1 timestep per channel.")

        if not timesteps:
            logger.warning(f"No timesteps found for channel {channel}. Skipping.")
            continue

        logger.debug(f"Timesteps for channel {channel}: {timesteps}")

        channel_name = f"channel_{channel:04d}"
        channel_path = output_dir / channel_name
        channel_path.mkdir(parents=True, exist_ok=True)

        channel_done_marker = channel_path / ".done"
        if channel_done_marker.exists():
            logger.info(
                f"Rank {fabric.global_rank}: Channel {channel} already processed. Skipping."
            )
            continue

        for timestep in timesteps:
            timestep_path = channel_path / f"timestep_{timestep:04d}"
            timestep_done_marker = timestep_path / ".done"
            if timestep_done_marker.exists():
                logger.info(
                    f"Rank {fabric.global_rank}: Channel {channel}, Timestep {timestep} already processed. Skipping."
                )
                continue

            try:
                latents, stats = generate_deep_dreams_for_channel_timestep(
                    stage_config=stage_config,
                    model_wrapper=model_wrapper,
                    channel=channel,
                    timestep=timestep,
                    activation_type=activation_type,
                    priors=priors,
                    output_dir=timestep_path,
                )

                # Final Save
                logger.debug(f"Saving final results for channel {channel}, timestep {timestep}")
                latents = latents.detach()
                images = model_wrapper.decode_latents(latents)
                latents = latents.cpu()

                with open(timestep_path / "stats.json", "w") as f:
                    json.dump(stats, f, indent=4)

                latents_dir = timestep_path / "latents"
                latents_dir.mkdir(parents=True, exist_ok=True)

                images_dir = timestep_path / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                for j, (image, latent) in enumerate(zip(images, latents)):
                    image = (image * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image)
                    image_pil.save(images_dir / f"deep_dream_image_{j:04d}.png")

                    save_file(
                        {"latent": latent}, latents_dir / f"deep_dream_latent_{j:04d}.safetensors"
                    )

                timestep_done_marker.touch()
            except Exception as e:
                logger.exception(f"Error processing channel {channel}, timestep {timestep}: {e}")

        if i % stage_config.log_every_n_steps == 0:
            current_time = time.time()
            elapsed_seconds = current_time - start_time
            batches_processed = i + 1

            avg_seconds_per_batch = (
                elapsed_seconds / batches_processed if batches_processed > 0 else 0
            )

            todo_for_one = int(len(channels_dataset) / fabric.world_size)
            remaining_batches = todo_for_one - batches_processed
            eta_seconds = remaining_batches * avg_seconds_per_batch
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                f"Rank {fabric.global_rank}: Processed {batches_processed}/{todo_for_one} channels. ETA: {eta_str}"
            )

        channel_done_marker.touch()

    logger.info(f"Rank {fabric.global_rank}: Done processing dataset! (sae: {sae})")
    fabric.barrier()


def run_deep_dream(config: ExperimentConfig):
    logger.info("Starting Fabric...")
    fabric = Fabric(accelerator=config.fabric.accelerator, devices="auto", strategy="ddp")
    fabric.launch()

    setup_distributed_logging(fabric.global_rank)
    logger.info(
        f"Fabric launched. Global Rank: {fabric.global_rank}, World Size: {fabric.world_size}"
    )

    stage_config = cast(DeepDreamStageConfig, config.stage_config)
    sae_stage_config = resolve_sae_config(stage_config)
    use_sae = config.use_sae

    logger.info(f"Stage Config - Use Prior: {stage_config.use_prior}, Use SAE: {use_sae}")

    if not stage_config.use_prior:
        if stage_config.seeds is None:
            logger.error("seeds must be provided in stage_config when use_prior is False")
            raise ValueError("seeds must be provided in stage_config when use_prior is False")

    timesteps_analysis_results_abs_path = (
        config.outputs_dir / stage_config.timestep_analysis_results_dir
    )
    prior_results_abs_path = config.outputs_dir / stage_config.prior_results_dir

    logger.info(f"Timestep Analysis Path: {timesteps_analysis_results_abs_path}")
    logger.info(f"Prior Results Path: {prior_results_abs_path}")

    try:
        generate_deep_dreams(
            config=config,
            stage_config=stage_config,
            timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
            prior_results_abs_path=prior_results_abs_path,
            sae=False,
            fabric=fabric,
        )

        if use_sae:
            generate_deep_dreams(
                config=config,
                stage_config=sae_stage_config,
                timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
                prior_results_abs_path=prior_results_abs_path,
                sae=True,
                fabric=fabric,
            )
    except Exception as e:
        logger.exception(f"Fatal error in run_deep_dream: {e}")
        raise

    logger.info("Deep Dream stage completed successfully!")
