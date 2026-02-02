import datetime
import json
from pathlib import Path
import time
from typing import cast

from diffusers import StableDiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]
from lightning import Fabric
from loguru import logger
from safetensors.torch import save_file
from submodules.SAeUron.SAE.sae import Sae
import torch
from torch.utils.data import DataLoader

from diffusion_deep_dream_research.config.config_schema import CaptureStageConfig, ExperimentConfig
from diffusion_deep_dream_research.core.data.unique_prompt_dataset import UniquePromptDataset
from diffusion_deep_dream_research.core.model.hooked_model_wrapper import HookedModelWrapper
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging
from diffusion_deep_dream_research.utils.torch_utils import get_dtype


def run_capture(config: ExperimentConfig):
    logger.info("Starting Fabric...")
    fabric = Fabric(
        accelerator=config.fabric.accelerator,
        devices="auto",
        strategy="ddp"
    )
    fabric.launch()

    setup_distributed_logging(fabric.global_rank)

    stage_conf = cast(CaptureStageConfig, config.stage_config)

    logger.info(f"Using dataset: {stage_conf.prompt_dataset.name}")
    dataset = UniquePromptDataset(stage_conf.prompt_dataset.path)
    logger.info(f"Loaded a prompt dataset of length: {len(dataset)}")

    logger.info(f"Using {stage_conf.num_images_per_prompt} images per prompt.")
    if stage_conf.dev_n_prompts is not None:
        logger.info(f"Running in development mode. Only processing {stage_conf.dev_n_prompts} prompts.")

    data_loader = DataLoader(
        dataset,
        batch_size=stage_conf.batch_size,
        shuffle=False,
        num_workers=stage_conf.num_workers,
        drop_last=False
    )

    data_loader = fabric.setup_dataloaders(data_loader)
    dtype = get_dtype()

    total_batches = len(data_loader)
    logger.info(f"Rank {fabric.global_rank}: Total batches to process on this rank: {total_batches}. World size: {fabric.world_size}")

    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_to_analyse.path,
        torch_dtype=dtype,
    )
    pipe.to(fabric.device)

    # Disable the progress bar for non-local runs
    pipe.set_progress_bar_config(disable=config.infrastructure_name != "local")

    if config.use_sae:
        #NOTE: SAE config is very SAeUron specific at the moment. Could be made more general
        sae_path = f"{config.sae.path}/unet.{config.target_layer_name}"
        sae = Sae.load_from_disk(
            path=sae_path,
            device=fabric.device,
            decoder=True
        )

        model_wrapper = HookedModelWrapper.from_sae(
            pipe=pipe,
            target_layer_name=config.target_layer_name,
            sae=sae
        )
    else:
        model_wrapper = HookedModelWrapper.from_layer(
            pipe=pipe,
            target_layer_name=config.target_layer_name,
        )

    # we are already in config.outputs_dir
    rank_dir = Path(f"fabric_rank_{fabric.global_rank}")
    rank_dir.mkdir(parents=True, exist_ok=True)

    model_wrapper.eval()

    start_time = time.time()

    with torch.no_grad():
        for i, batch_prompts in enumerate(data_loader):
            if stage_conf.dev_n_prompts is not None:
                if i >= stage_conf.dev_n_prompts:
                    break

            batch_dir = rank_dir / f"batch_{i:05d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            done_marker = batch_dir / ".done"

            if done_marker.exists():
                if i % stage_conf.log_every_n_steps == 0:
                    logger.info(f"Rank {fabric.global_rank}: Batch {i} already completed. Skipping.")
                continue

            prompts = list(batch_prompts)

            # Decided to keep randomness without defining seeds
            # I think it ensures better coverage and variance
            # The seed should not be affecting the timestep localization of activity
            # That's my intuition at least.
            # Models are trained on variable noise and should be resilient to it
            result = model_wrapper.forward_with_capture(
                prompts=prompts, # batch_size
                num_images_per_prompt=stage_conf.num_images_per_prompt
            )
            # result.hook_activations: Dict[timestep, Tensor[batch_size * num_images_per_prompt, channels]]
            # result.images: Tensor -> [prompt_1_image_1, prompt_1_image_2, ...]

            prompts_repeated = [p for p in prompts for _ in range(stage_conf.num_images_per_prompt)]
            with open(batch_dir / f"prompts.json", "w") as f:
                json.dump(prompts_repeated, f)

            images_save_path = batch_dir / "generated_images"
            images_save_path.mkdir(parents=True, exist_ok=True)
            for img_idx, image in enumerate(result.images):
                image.save(images_save_path / f"image_{img_idx:04d}.png")


            for timestep, actvations_per_type in result.hook_activations.items():
                timestep_path = batch_dir / f"timestep_{timestep:04d}"
                timestep_path.mkdir(parents=True, exist_ok=True)
                for act_type, activations in actvations_per_type.items():
                    # Dict[timestep, Dict[ActivationType, Tensor (batch_size, channels)]]
                    save_path = timestep_path / f"capture_{act_type.value}.safetensors"
                    save_file({"activations": activations}, save_path)


            done_marker.touch()

            if i % stage_conf.log_every_n_steps == 0:
                current_time = time.time()
                elapsed_seconds = current_time - start_time
                batches_processed = i + 1
                
                avg_seconds_per_batch = elapsed_seconds / batches_processed if batches_processed > 0 else 0
                
                remaining_batches = total_batches - batches_processed
                eta_seconds = remaining_batches * avg_seconds_per_batch

                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                logger.info(f"Rank {fabric.global_rank}: Processed {batches_processed}/{total_batches} batches. ETA: {eta_str}")

    fabric.barrier()