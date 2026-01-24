from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
from safetensors.torch import load_file


@dataclass
class Activations:
    encoded: Optional[np.ndarray]  # (total_batch_size, sae_channel)
    raw: np.ndarray  # (total_batch_size, channel)


@dataclass
class Batch:
    prompts: list[str]
    generated_image_paths: list[Path]
    activations_per_timestep: dict[int, Activations]


def get_batches(path: Path) -> list[Batch]:
    batches: list[Batch] = []

    all_rank_dirs = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("fabric_rank_")]
    )

    logger.info(f"Found {len(all_rank_dirs)} rank directories.")
    for rank_dir in all_rank_dirs:
        logger.debug(f"Processing rank directory: {rank_dir}")

        all_batch_dirs = sorted(
            [d for d in rank_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
        )

        logger.debug(f"Found {len(all_batch_dirs)} batch directories.")

        for batch_dir in all_batch_dirs:
            logger.debug(f"Processing batch directory: {batch_dir}")
            images_dir = batch_dir / "generated_images"
            if not images_dir.exists():
                raise ValueError(f"generated_images directory does not exist in {images_dir}")

            generated_image_paths = sorted(
                [f for f in images_dir.iterdir() if f.is_file() and f.name.startswith("image_")]
            )
            logger.debug(f"Found {len(generated_image_paths)} generated images.")

            prompt_file = batch_dir / "prompts.json"
            if not prompt_file.exists():
                raise ValueError(f"prompts.json does not exist in {prompt_file}")

            with open(prompt_file, "r") as f:
                prompts = json.load(f)
            logger.debug(f"Loaded {len(prompts)} prompts.")

            all_timestep_dirs = sorted(
                [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("timestep_")]
            )
            logger.debug(f"Found {len(all_timestep_dirs)} timestep directories.")

            activations_per_timestep: dict[int, Activations] = {}

            for timestep_dir in all_timestep_dirs:
                logger.debug(f"Processing timestep directory: {timestep_dir}")
                timestep_num = int(timestep_dir.name.split("_")[1])

                activations_encoded_path = timestep_dir / "capture_encoded.safetensors"
                activations_raw_path = timestep_dir / "capture_raw.safetensors"

                # SAE activations are optional
                if activations_encoded_path.exists():
                    activations_encoded = (
                        load_file(activations_encoded_path)["activations"].detach().cpu().numpy()
                    )
                else:
                    activations_encoded = None

                if not activations_raw_path.exists():
                    raise ValueError(
                        f"capture_raw.safetensors does not exist in {activations_raw_path}"
                    )

                activations_raw = (
                    load_file(activations_raw_path)["activations"].detach().cpu().numpy()
                )

                activations = Activations(encoded=activations_encoded, raw=activations_raw)

                activations_per_timestep[timestep_num] = activations

            batch = Batch(
                prompts=prompts,
                generated_image_paths=generated_image_paths,
                activations_per_timestep=activations_per_timestep,
            )

            batches.append(batch)

    return batches
