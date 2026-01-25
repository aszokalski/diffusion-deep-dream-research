from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image
from safetensors import safe_open
import torch


@dataclass
class ImageWithLatent:
    image_path: Path
    latent_path: Path

    @property
    def image(self) -> Image.Image:
        logger.debug(f"Opening image: {self.image_path}")
        return Image.open(self.image_path)

    @property
    def latent(self) -> torch.Tensor:
        """
        Returns a CPU tensor of the latent
        """
        logger.debug(f"Loading latent tensor from: {self.latent_path}")
        try:
            with safe_open(self.latent_path, framework="pt", device="cpu") as f:
                key = list(f.keys())[0]
                latent = f.get_tensor(key)
                logger.trace(f"Loaded latent shape: {latent.shape}")
            return latent
        except Exception as e:
            logger.error(f"Failed to load latent from {self.latent_path}: {e}")
            raise


@dataclass
class ChannelPriors:
    images_with_latents: list[ImageWithLatent]

    def get_latents(
        self, device: torch.device, dtype: torch.dtype, n_results=None
    ) -> torch.Tensor:
        """
        Returns a tensor of shape (num_images, latent_dim...) bound to CPU
        :return:
        """
        total_available = len(self.images_with_latents)
        if n_results is not None:
            logger.debug(f"Requesting {n_results} latents (available: {total_available})")
            subset = self.images_with_latents[:n_results]
        else:
            logger.debug(f"Requesting all {total_available} latents")
            subset = self.images_with_latents

        logger.debug("Stacking latent tensors...")
        latents_list = [iwl.latent for iwl in subset]

        if not latents_list:
            logger.warning("No latents found to stack. Returning empty tensor.")
            return torch.empty(0, device=device, dtype=dtype)

        latents = torch.stack(latents_list, dim=0)
        logger.debug(f"Stacked tensor shape: {latents.shape}. Moving to {device} as {dtype}.")

        return latents.detach().to(device=device, dtype=dtype)


@dataclass
class PriorResults:
    raw: dict[int, ChannelPriors]
    sae: dict[int, ChannelPriors]


def _parse_channel_dir(channel_path: Path) -> list[ImageWithLatent]:
    """Helper to pair images and latents within a channel directory."""
    logger.debug(f"Parsing channel directory: {channel_path}")

    images_dir = channel_path / "images"
    latents_dir = channel_path / "latents"

    results = []

    if not images_dir.exists():
        logger.warning(f"Images directory missing at {images_dir}")
        return results
    if not latents_dir.exists():
        logger.warning(f"Latents directory missing at {latents_dir}")
        return results

    # Get all image files first
    image_files = sorted(list(images_dir.glob("prior_image_*.png")))
    logger.debug(f"Found {len(image_files)} image files in {images_dir}")

    for image_path in image_files:
        try:
            file_id = image_path.stem.split("_")[-1]
            latent_path = latents_dir / f"prior_latent_{file_id}.safetensors"

            if latent_path.exists():
                results.append(ImageWithLatent(image_path=image_path, latent_path=latent_path))
            else:
                logger.error(
                    f"Latent file missing for image {image_path}. Expected: {latent_path}"
                )
                raise ValueError(f"Latent file missing for image {image_path}")
        except Exception as e:
            logger.error(f"Error parsing pair for {image_path}: {e}")
            raise

    logger.debug(f"Successfully paired {len(results)} items in {channel_path.name}")
    return results


def get_prior_results(root_path: Path) -> PriorResults:
    """
    Reads prior results from the given root path.
    """
    logger.info(f"Scanning for prior results in: {root_path}")

    raw_channels: dict[int, ChannelPriors] = {}
    sae_channels: dict[int, ChannelPriors] = {}

    rank_dirs = sorted(root_path.glob("fabric_rank_*"))
    logger.info(f"Found {len(rank_dirs)} fabric rank directories.")

    for rank_dir in rank_dirs:
        logger.debug(f"Processing rank directory: {rank_dir}")

        # Process Raw Priors
        priors_dir = rank_dir / "priors"
        if priors_dir.exists():
            channel_dirs = list(priors_dir.glob("channel_*"))
            logger.debug(f"Found {len(channel_dirs)} raw channel directories in {priors_dir}")

            for channel_dir in channel_dirs:
                try:
                    channel_id = int(channel_dir.name.split("_")[-1])
                    pairs = _parse_channel_dir(channel_dir)
                    if pairs:
                        if channel_id in raw_channels:
                            logger.warning(
                                f"Duplicate channel ID {channel_id} found in {rank_dir}. Skipping."
                            )
                            continue
                        else:
                            raw_channels[channel_id] = ChannelPriors(images_with_latents=pairs)
                except Exception as e:
                    logger.error(f"Failed to process raw channel dir {channel_dir}: {e}")

        # Process SAE Priors
        priors_sae_dir = rank_dir / "priors_sae"
        if priors_sae_dir.exists():
            channel_dirs_sae = list(priors_sae_dir.glob("channel_*"))
            logger.debug(
                f"Found {len(channel_dirs_sae)} SAE channel directories in {priors_sae_dir}"
            )

            for channel_dir in channel_dirs_sae:
                try:
                    channel_id = int(channel_dir.name.split("_")[-1])
                    pairs = _parse_channel_dir(channel_dir)
                    if pairs:
                        if channel_id in sae_channels:
                            logger.warning(
                                f"Duplicate SAE channel ID {channel_id} found in {rank_dir}. Skipping."
                            )
                            continue
                        else:
                            sae_channels[channel_id] = ChannelPriors(images_with_latents=pairs)
                except ValueError as e:
                    logger.error(f"Skipping invalid SAE channel dir {channel_dir}: {e}")
                    continue
                except Exception as e:
                    logger.exception(f"Unexpected error in SAE channel dir {channel_dir}: {e}")

    logger.info(
        f"Completed loading priors. Found {len(raw_channels)} raw channels and {len(sae_channels)} SAE channels."
    )
    return PriorResults(raw=raw_channels, sae=sae_channels)
