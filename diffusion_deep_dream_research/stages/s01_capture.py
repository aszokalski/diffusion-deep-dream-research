import json
from pathlib import Path
from typing import cast
from loguru import logger

import torch
from diffusers import StableDiffusionPipeline
from lightning import Fabric
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, CaptureStageConfig
from diffusion_deep_dream_research.core.data.unique_prompt_dataset import UniquePromptDataset
from diffusion_deep_dream_research.core.model.hooked_model_wrapper import HookedModelWrapper
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging
from diffusion_deep_dream_research.utils.torch_utils import get_dtype

from submodules.SAeUron.SAE.sae import Sae


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
                prompts=prompts,
                num_images_per_prompt=stage_conf.num_images_per_prompt
            )
            # result.hook_activations: Dict[timestep, Tensor[batch_size, channels]]
            # result.images: Tensor -> [prompt_1_image_1, prompt_1_image_2, ...]

            prompts_repeated = [p for p in prompts for _ in range(stage_conf.num_images_per_prompt)]
            with open(batch_dir / f"prompts.json", "w") as f:
                json.dump(prompts_repeated, f)

            images_save_path = batch_dir / "generated_images"
            images_save_path.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(result.images):
                #save pil images as png
                image.save(images_save_path / f"image_{i:04d}.png")


            for timestep, actvations_per_type in result.hook_activations.items():
                timestep_path = batch_dir / f"timestep_{timestep:04d}"
                timestep_path.mkdir(parents=True, exist_ok=True)
                for act_type, activations in actvations_per_type.items():
                    # Dict[timestep, Dict[ActivationType, Tensor (batch_size, channels)]]
                    save_path = timestep_path / f"capture_{act_type.value}.safetensors"
                    save_file({"activations": activations}, save_path)


            done_marker.touch()

            if i % stage_conf.log_every_n_steps == 0:
                logger.info(f"Rank {fabric.global_rank}: Processed {i+1} batches...")

    fabric.barrier()
    logger.info(f"Capture stage completed successfully. Results at: {config.outputs_dir}")