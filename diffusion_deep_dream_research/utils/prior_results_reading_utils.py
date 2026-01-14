from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
from safetensors import safe_open


@dataclass
class ImageWithLatent:
    image_path: Path
    latent_path: Path

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)

    @property
    def latent(self) -> torch.Tensor:
        """
        Returns a CPU tensor of the latent
        """
        with safe_open(self.latent_path, framework="pt", device="cpu") as f:
            latent = f.get_tensor(list(f.keys())[0])
        return latent


@dataclass
class ChannelPriors:
    channel_id: int
    images_with_latents: list[ImageWithLatent]

    def get_latents(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns a tensor of shape (num_images, latent_dim...) bound to CPU
        :return:
        """
        return torch.stack([iwl.latent for iwl in self.images_with_latents], dim=0).detach().to(device=device, dtype=dtype)


@dataclass
class PriorResults:
    raw: list[ChannelPriors]
    sae: list[ChannelPriors]


def _parse_channel_dir(channel_path: Path) -> list[ImageWithLatent]:
    """Helper to pair images and latents within a channel directory."""
    images_dir = channel_path / "images"
    latents_dir = channel_path / "latents"

    results = []

    if not images_dir.exists() or not latents_dir.exists():
        return results


    for image_path in sorted(images_dir.glob("prior_image_*.png")):
        file_id = image_path.stem.split('_')[-1]
        latent_path = latents_dir / f"prior_latent_{file_id}.safetensors"

        if latent_path.exists():
            results.append(ImageWithLatent(image_path=image_path, latent_path=latent_path))
        else:
            raise ValueError(f"Latent file missing for image {image_path}")

    return results


def get_prior_results(root_path: Path) -> PriorResults:
    """
    Reads prior results from the given root path.
    """
    raw_channels: dict[int, ChannelPriors] = {}
    sae_channels: dict[int, ChannelPriors] = {}

    rank_dirs = sorted(root_path.glob("fabric_rank_*"))

    for rank_dir in rank_dirs:

        priors_dir = rank_dir / "priors"

        if priors_dir.exists():
            for channel_dir in priors_dir.glob("channel_*"):
                channel_id = int(channel_dir.name.split('_')[-1])
                pairs = _parse_channel_dir(channel_dir)
                if pairs:
                    raw_channels[channel_id] = ChannelPriors(channel_id=channel_id, images_with_latents=pairs)

        priors_sae_dir = rank_dir / "priors_sae"
        if priors_sae_dir.exists():
            for channel_dir in priors_sae_dir.glob("channel_*"):
                try:
                    channel_id = int(channel_dir.name.split('_')[-1])
                    pairs = _parse_channel_dir(channel_dir)
                    if pairs:
                        sae_channels[channel_id] = ChannelPriors(channel_id=channel_id, images_with_latents=pairs)
                except ValueError:
                    continue

    # Sort lists by channel_id
    sorted_raw = [raw_channels[k] for k in sorted(raw_channels.keys())]
    sorted_sae = [sae_channels[k] for k in sorted(sae_channels.keys())]

    return PriorResults(
        raw=sorted_raw,
        sae=sorted_sae
    )