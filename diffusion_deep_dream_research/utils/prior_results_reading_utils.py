import concurrent.futures
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from loguru import logger
from PIL import Image
from safetensors import safe_open
import torch

from diffusion_deep_dream_research.utils.path_utils import extract_id


class PriorMapData(TypedDict):
    images: Dict[int, Path]
    latents: Dict[int, Path]


@dataclass
class ImageWithLatent:
    image_path: Path
    latent_path: Path

    @property
    def image(self) -> Image.Image:
        return Image.open(self.image_path)

    @property
    def latent(self) -> torch.Tensor:
        """Returns a CPU tensor of the latent."""
        try:
            with safe_open(self.latent_path, framework="pt", device="cpu") as f:
                return f.get_tensor(list(f.keys())[0])
        except Exception as e:
            logger.error(f"Failed to load latent from {self.latent_path}: {e}")
            raise


@dataclass
class ChannelPriors:
    images_with_latents: List[ImageWithLatent]

    def get_latents(
        self, device: torch.device, dtype: torch.dtype, n_results=None
    ) -> torch.Tensor:
        """Returns a tensor of shape (num_images, latent_dim...) bound to CPU."""
        subset = (
            self.images_with_latents[:n_results]
            if n_results is not None
            else self.images_with_latents
        )

        latents_list = [iwl.latent for iwl in subset]

        if not latents_list:
            logger.warning("No latents found to stack. Returning empty tensor.")
            return torch.empty(0, device=device, dtype=dtype)

        return torch.stack(latents_list, dim=0).detach().to(device=device, dtype=dtype)


@dataclass
class PriorResults:
    raw: Dict[int, ChannelPriors]
    sae: Dict[int, ChannelPriors]


def _scan_channel_files(channel_path: Path) -> PriorMapData:
    data_map: PriorMapData = {"images": {}, "latents": {}}

    for root, _, files in os.walk(channel_path):
        if not files:
            continue

        root_path = Path(root)
        parent_name = root_path.name

        if parent_name == "images":
            for f in files:
                if f.startswith("prior_image_") and f.endswith(".png"):
                    if (idx := extract_id(f)) is not None:
                        data_map["images"][idx] = root_path / f

        elif parent_name == "latents":
            for f in files:
                if f.startswith("prior_latent_") and f.endswith(".safetensors"):
                    if (idx := extract_id(f)) is not None:
                        data_map["latents"][idx] = root_path / f

    return data_map


def _assemble_channel_priors(data_map: PriorMapData) -> Optional[List[ImageWithLatent]]:
    results = []

    sorted_indices = sorted(data_map["images"].keys())

    for idx in sorted_indices:
        if idx in data_map["latents"]:
            results.append(
                ImageWithLatent(
                    image_path=data_map["images"][idx], latent_path=data_map["latents"][idx]
                )
            )

    return results if results else None


def _process_channel(channel_path: Path) -> Optional[Tuple[int, List[ImageWithLatent]]]:
    cid = extract_id(channel_path.name)
    if cid is None:
        return None

    raw_map = _scan_channel_files(channel_path)

    priors_list = _assemble_channel_priors(raw_map)

    if not priors_list:
        return None

    return cid, priors_list


def get_prior_results(root_path: Path) -> PriorResults:
    logger.info(f"Scanning for prior results in: {root_path}")

    if not root_path.exists():
        logger.error(f"Root path does not exist: {root_path}")
        return PriorResults(raw={}, sae={})

    raw_channels: Dict[int, ChannelPriors] = {}
    sae_channels: Dict[int, ChannelPriors] = {}

    tasks = []
    for p in root_path.glob("fabric_rank_*/priors/channel_*"):
        tasks.append(("raw", p))
    for p in root_path.glob("fabric_rank_*/priors_sae/channel_*"):
        tasks.append(("sae", p))

    logger.info(f"Detected {len(tasks)} channels to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        future_map = {executor.submit(_process_channel, p): r_type for r_type, p in tasks}

        for i, future in enumerate(concurrent.futures.as_completed(future_map)):
            r_type = future_map[future]
            try:
                res = future.result()
                if res:
                    cid, pairs = res
                    target_dict = raw_channels if r_type == "raw" else sae_channels

                    if cid not in target_dict:
                        target_dict[cid] = ChannelPriors(images_with_latents=pairs)

            except Exception as e:
                logger.error(f"Worker failed: {e}")

            if (i + 1) % 2000 == 0:
                logger.info(f"Progress: {i + 1}/{len(tasks)} channels processed...")

    logger.success(f"Completed loading priors. Raw: {len(raw_channels)}, SAE: {len(sae_channels)}")
    return PriorResults(raw=raw_channels, sae=sae_channels)
