from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from loguru import logger
from PIL import Image
from safetensors import safe_open
import torch

NoiseMode = Literal["noise", "no_noise"]


@dataclass
class DeepDreamStats:
    step: int
    activation: float
    penalties: Dict[str, float]
    total_loss: float


@dataclass
class IntermediateStep:
    step_idx: int
    image_path: Path  # Changed: Single path for this specific variant
    stats_path: Path

    @property
    def stats(self) -> DeepDreamStats:
        logger.debug(f"Reading intermediate stats from {self.stats_path}")
        with open(self.stats_path, "r") as f:
            data = json.load(f)
        return DeepDreamStats(
            step=data["step"],
            activation=data["activation"],
            penalties=data["penalties"],
            total_loss=data["total_loss"],
        )

    def get_image(self) -> Image.Image:
        logger.debug(f"Opening intermediate image from {self.image_path}")
        return Image.open(self.image_path)


@dataclass
class DeepDreamResult:
    variant_idx: int  # Added: To track which batch index this was (0, 1, 2...)
    final_image_path: Path
    final_latent_path: Path
    stats_path: Path
    intermediate_steps: List[IntermediateStep]

    @property
    def stats(self) -> DeepDreamStats:
        """Returns the final stats for the optimization run (shared across variants)."""
        logger.debug(f"Reading final stats from {self.stats_path}")
        with open(self.stats_path, "r") as f:
            data = json.load(f)
        return DeepDreamStats(
            step=data["step"],
            activation=data["activation"],
            penalties=data["penalties"],
            total_loss=data["total_loss"],
        )

    def get_final_image(self) -> Image.Image:
        logger.debug(f"Opening final image from {self.final_image_path}")
        return Image.open(self.final_image_path)

    def get_final_latent(self, device="cpu") -> torch.Tensor:
        logger.debug(f"Loading final latent from {self.final_latent_path} on {device}")
        with safe_open(self.final_latent_path, framework="pt", device=device) as f:
            key = list(f.keys())[0]
            return f.get_tensor(key)


# Changed: The leaf value is now a List of results instead of a single object
TimestepDict = Dict[int, Dict[NoiseMode, List[DeepDreamResult]]]
ChannelDict = Dict[int, TimestepDict]


def _parse_timestep_dir(timestep_path: Path) -> List[DeepDreamResult]:
    logger.debug(f"Parsing timestep directory: {timestep_path}")
    images_dir = timestep_path / "images"
    latents_dir = timestep_path / "latents"
    intermediate_dir = timestep_path / "intermediate"
    stats_path = timestep_path / "stats.json"

    results: List[DeepDreamResult] = []

    if not images_dir.exists():
        logger.warning(f"Images directory not found at {images_dir}")
        return results

    # 1. Identify all variants by looking at the final images
    # Pattern: deep_dream_image_{idx:04d}.png
    final_image_files = sorted(images_dir.glob("deep_dream_image_*.png"))
    logger.debug(f"Found {len(final_image_files)} final image files in {images_dir}")

    for img_path in final_image_files:
        try:
            # Extract index (e.g. "0000" from "deep_dream_image_0000.png")
            idx_str = img_path.stem.split("_")[-1]
            idx = int(idx_str)
        except ValueError as e:
            logger.error(f"Failed to parse index from image path {img_path}: {e}")
            continue

        logger.debug(f"Processing variant index {idx} (idx_str: {idx_str})")

        # 2. Find corresponding latent
        latent_path = latents_dir / f"deep_dream_latent_{idx_str}.safetensors"
        if not latent_path.exists():
            logger.warning(f"Latent file missing for variant {idx}: {latent_path}")

        # 3. Find specific intermediate steps for this variant index
        variant_steps: List[IntermediateStep] = []
        if intermediate_dir.exists():
            # Iterate through step directories (step_0000, step_0010, etc.)
            step_dirs = sorted(intermediate_dir.glob("step_*"))
            # logger.trace(f"Found {len(step_dirs)} step directories in {intermediate_dir}")

            for step_dir in step_dirs:
                try:
                    step_idx = int(step_dir.name.split("_")[-1])
                    step_stats_path = step_dir / "stats.json"

                    # Look for the specific image for this variant inside the step folder
                    # e.g., intermediate/step_0010/image_0000.png
                    step_img_path = step_dir / f"image_{idx_str}.png"

                    if step_img_path.exists() and step_stats_path.exists():
                        variant_steps.append(
                            IntermediateStep(
                                step_idx=step_idx,
                                image_path=step_img_path,
                                stats_path=step_stats_path,
                            )
                        )
                    else:
                        # Optional: Log if intermediate file is missing (might be verbose)
                        pass
                except ValueError as e:
                    logger.warning(f"Failed to parse step index from directory {step_dir}: {e}")
                    continue
        else:
            logger.debug(f"Intermediate directory not found at {intermediate_dir}")

        # Sort steps just in case
        variant_steps.sort(key=lambda x: x.step_idx)
        logger.debug(f"Found {len(variant_steps)} intermediate steps for variant {idx}")

        # 4. Create the result object for this specific variant
        if latent_path.exists():
            results.append(
                DeepDreamResult(
                    variant_idx=idx,
                    final_image_path=img_path,
                    final_latent_path=latent_path,
                    stats_path=stats_path,
                    intermediate_steps=variant_steps,
                )
            )
            logger.debug(f"Successfully added DeepDreamResult for variant {idx}")
        else:
            logger.warning(f"Skipping variant {idx} due to missing latent path")

    # Sort results by variant index (0, 1, 2...)
    sorted_results = sorted(results, key=lambda x: x.variant_idx)
    logger.info(f"Returning {len(sorted_results)} sorted results from {timestep_path}")
    return sorted_results


def _process_root_directory(
    root_path: Path, noise_mode: NoiseMode, raw_results: ChannelDict, sae_results: ChannelDict
) -> None:
    logger.info(f"Processing root directory: {root_path} with noise_mode='{noise_mode}'")
    if not root_path or not root_path.exists():
        logger.error(f"Root path does not exist: {root_path}")
        return

    rank_dirs = list(root_path.glob("fabric_rank_*"))
    logger.info(f"Found {len(rank_dirs)} rank directories in {root_path}")

    for rank_dir in rank_dirs:
        logger.debug(f"Entering rank directory: {rank_dir}")

        # Process Standard Deep Dream
        deep_dream_dir = rank_dir / "deep_dream"
        if deep_dream_dir.exists():
            logger.debug(f"Found standard deep_dream directory: {deep_dream_dir}")
            channel_dirs = list(deep_dream_dir.glob("channel_*"))
            logger.info(f"Found {len(channel_dirs)} channels in standard deep_dream")

            for channel_dir in channel_dirs:
                try:
                    channel_id = int(channel_dir.name.split("_")[-1])
                except ValueError as e:
                    logger.warning(f"Skipping invalid channel dir {channel_dir}: {e}")
                    continue

                if channel_id not in raw_results:
                    raw_results[channel_id] = {}
                    logger.debug(f"Initialized raw_results entry for channel {channel_id}")

                timestep_dirs = list(channel_dir.glob("timestep_*"))
                # logger.debug(f"Found {len(timestep_dirs)} timesteps for channel {channel_id}")

                for timestep_dir in timestep_dirs:
                    try:
                        timestep = int(timestep_dir.name.split("_")[-1])
                    except ValueError as e:
                        logger.warning(f"Skipping invalid timestep dir {timestep_dir}: {e}")
                        continue

                    if timestep not in raw_results[channel_id]:
                        raw_results[channel_id][timestep] = {}

                    logger.debug(f"Parsing standard timestep {timestep} for channel {channel_id}")
                    raw_results[channel_id][timestep][noise_mode] = _parse_timestep_dir(
                        timestep_dir
                    )
        else:
            logger.debug(f"No standard deep_dream directory found in {rank_dir}")

        # Process SAE Deep Dream
        deep_dream_sae_dir = rank_dir / "deep_dream_sae"
        if deep_dream_sae_dir.exists():
            logger.debug(f"Found SAE deep_dream directory: {deep_dream_sae_dir}")
            channel_dirs_sae = list(deep_dream_sae_dir.glob("channel_*"))
            logger.info(f"Found {len(channel_dirs_sae)} channels in SAE deep_dream")

            for channel_dir in channel_dirs_sae:
                try:
                    channel_id = int(channel_dir.name.split("_")[-1])
                except ValueError as e:
                    logger.warning(f"Skipping invalid SAE channel dir {channel_dir}: {e}")
                    continue

                if channel_id not in sae_results:
                    sae_results[channel_id] = {}
                    logger.debug(f"Initialized sae_results entry for channel {channel_id}")

                timestep_dirs_sae = list(channel_dir.glob("timestep_*"))
                # logger.debug(f"Found {len(timestep_dirs_sae)} timesteps for SAE channel {channel_id}")

                for timestep_dir in timestep_dirs_sae:
                    try:
                        timestep = int(timestep_dir.name.split("_")[-1])
                    except ValueError as e:
                        logger.warning(f"Skipping invalid SAE timestep dir {timestep_dir}: {e}")
                        continue

                    if timestep not in sae_results[channel_id]:
                        sae_results[channel_id][timestep] = {}

                    logger.debug(f"Parsing SAE timestep {timestep} for channel {channel_id}")
                    sae_results[channel_id][timestep][noise_mode] = _parse_timestep_dir(
                        timestep_dir
                    )
        else:
            logger.debug(f"No SAE deep_dream directory found in {rank_dir}")


def get_deep_dream_results(
    noise_dir: Path, no_noise_dir: Optional[Path]
) -> tuple[ChannelDict, ChannelDict]:
    logger.info("Starting get_deep_dream_results")
    raw_channels: ChannelDict = {}
    sae_channels: ChannelDict = {}

    logger.info(f"Processing noise directory: {noise_dir}")
    _process_root_directory(noise_dir, "noise", raw_channels, sae_channels)

    if no_noise_dir:
        logger.info(f"Processing no_noise directory: {no_noise_dir}")
        _process_root_directory(no_noise_dir, "no_noise", raw_channels, sae_channels)

    logger.success(
        f"Completed loading results. Raw Channels: {len(raw_channels)}, SAE Channels: {len(sae_channels)}"
    )
    return raw_channels, sae_channels
