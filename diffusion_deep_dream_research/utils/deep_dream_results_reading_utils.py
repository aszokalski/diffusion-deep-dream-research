import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

from loguru import logger
from PIL import Image
from safetensors import safe_open
import torch

from diffusion_deep_dream_research.utils.path_utils import extract_id

NoiseMode = Literal["noise", "no_noise"]


class ChannelMapData(TypedDict):
    images: Dict[int, Path]
    latents: Dict[int, Path]
    intermediate: List[
        Tuple[int, int, Path, Path]
    ]  # (variant_idx, step_idx, img_path, stats_path)
    stats: Optional[Path]


TimestepMap = Dict[int, ChannelMapData]


@dataclass
class DeepDreamStats:
    step: int
    activation: float
    penalties: Dict[str, float]
    total_loss: float


@dataclass
class IntermediateStep:
    step_idx: int
    image_path: Path
    stats_path: Path

    @property
    def stats(self) -> DeepDreamStats:
        return _load_stats(self.stats_path)

    def get_image(self) -> Image.Image:
        return Image.open(self.image_path)


@dataclass
class DeepDreamResult:
    variant_idx: int
    final_image_path: Path
    final_latent_path: Path
    stats_path: Path
    intermediate_steps: List[IntermediateStep]

    @property
    def stats(self) -> DeepDreamStats:
        return _load_stats(self.stats_path)

    def get_final_image(self) -> Image.Image:
        return Image.open(self.final_image_path)

    def get_final_latent(self, device="cpu") -> torch.Tensor:
        with safe_open(self.final_latent_path, framework="pt", device=device) as f:
            return f.get_tensor(list(f.keys())[0])


def _load_stats(path: Path) -> DeepDreamStats:
    with open(path, "r") as f:
        data = json.load(f)
    return DeepDreamStats(
        step=data["step"],
        activation=data["activation"],
        penalties=data["penalties"],
        total_loss=data["total_loss"],
    )


def _get_timestep_id(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        if part.startswith("timestep_"):
            return int(part.split("_")[-1])
    return None


def _scan_channel_files(channel_path: Path) -> TimestepMap:
    data_map: TimestepMap = {}

    for root, _, files in os.walk(channel_path):
        if not files:
            continue

        root_path = Path(root)

        if "timestep_" not in str(root_path):
            continue

        timestep = _get_timestep_id(root_path)
        if timestep is None:
            continue

        if timestep not in data_map:
            data_map[timestep] = {"images": {}, "latents": {}, "intermediate": [], "stats": None}

        entry = data_map[timestep]
        parent_name = root_path.name
        has_stats = "stats.json" in files

        if parent_name == "images":
            for f in files:
                if f.startswith("deep_dream_image_") and f.endswith(".png"):
                    if (idx := extract_id(f)) is not None:
                        entry["images"][idx] = root_path / f

        elif parent_name == "latents":
            for f in files:
                if f.startswith("deep_dream_latent_") and f.endswith(".safetensors"):
                    if (idx := extract_id(f)) is not None:
                        entry["latents"][idx] = root_path / f

        elif has_stats and parent_name.startswith("timestep_"):
            entry["stats"] = root_path / "stats.json"

        elif parent_name.startswith("step_"):
            step_idx = extract_id(parent_name)  # Extract '10' from 'step_10'
            if step_idx is not None:
                for f in files:
                    if f.startswith("image_") and f.endswith(".png"):
                        if (idx := extract_id(f)) is not None:
                            # Only add if stats exist (as per original logic)
                            if has_stats:
                                entry["intermediate"].append(
                                    (idx, step_idx, root_path / f, root_path / "stats.json")
                                )
    return data_map


def _assemble_channel_results(
    data_map: TimestepMap, noise_mode: NoiseMode
) -> Dict[int, Dict[NoiseMode, List[DeepDreamResult]]]:
    final_timesteps_map = {}

    for timestep, content in data_map.items():
        results_list = []

        for idx, img_path in content["images"].items():
            latent_path = content["latents"].get(idx)
            if not latent_path:
                continue

            stats_path = content["stats"] or (img_path.parent.parent / "stats.json")

            variant_steps = [
                IntermediateStep(step_idx=s_idx, image_path=i_path, stats_path=s_path)
                for v_idx, s_idx, i_path, s_path in content["intermediate"]
                if v_idx == idx
            ]
            variant_steps.sort(key=lambda x: x.step_idx)

            results_list.append(
                DeepDreamResult(
                    variant_idx=idx,
                    final_image_path=img_path,
                    final_latent_path=latent_path,
                    stats_path=stats_path,
                    intermediate_steps=variant_steps,
                )
            )

        if results_list:
            results_list.sort(key=lambda x: x.variant_idx)
            final_timesteps_map[timestep] = {noise_mode: results_list}

    return final_timesteps_map


def _process_channel(channel_path: Path, noise_mode: NoiseMode):
    cid = extract_id(channel_path.name)
    if cid is None:
        return None

    raw_map = _scan_channel_files(channel_path)

    results = _assemble_channel_results(raw_map, noise_mode)

    return cid, results


def _process_root_directory(
    root_path: Path, noise_mode: NoiseMode, raw_results: dict, sae_results: dict
) -> None:
    logger.info(f"Scanning root directory: {root_path}")
    if not root_path.exists():
        logger.error(f"Root path does not exist: {root_path}")
        return

    tasks = []
    for p in root_path.glob("fabric_rank_*/deep_dream/channel_*"):
        tasks.append(("raw", p))
    for p in root_path.glob("fabric_rank_*/deep_dream_sae/channel_*"):
        tasks.append(("sae", p))

    logger.info(f"Detected {len(tasks)} channels to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        future_map = {
            executor.submit(_process_channel, p, noise_mode): r_type for r_type, p in tasks
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_map)):
            r_type = future_map[future]
            try:
                res = future.result()
                if res:
                    cid, t_data = res
                    target_dict = raw_results if r_type == "raw" else sae_results

                    channel_entry = target_dict.setdefault(cid, {})
                    for t, modes in t_data.items():
                        channel_entry.setdefault(t, {}).update(modes)

            except Exception as e:
                logger.error(f"Error processing channel: {e}")

            if (i + 1) % 2000 == 0:
                logger.info(f"Progress: {i + 1}/{len(tasks)} channels processed...")


def get_deep_dream_results(noise_dir: Path, no_noise_dir: Optional[Path]) -> tuple[dict, dict]:
    logger.info("Starting deep dream results indexing...")
    raw_channels, sae_channels = {}, {}

    if noise_dir:
        _process_root_directory(noise_dir, "noise", raw_channels, sae_channels)
    if no_noise_dir:
        _process_root_directory(no_noise_dir, "no_noise", raw_channels, sae_channels)

    logger.success(f"Indexing Complete. Raw: {len(raw_channels)}, SAE: {len(sae_channels)}")
    return raw_channels, sae_channels
