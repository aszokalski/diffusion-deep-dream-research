from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Literal

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
        with open(self.stats_path, "r") as f:
            data = json.load(f)
        return DeepDreamStats(
            step=data["step"],
            activation=data["activation"],
            penalties=data["penalties"],
            total_loss=data["total_loss"],
        )

    def get_image(self) -> Image.Image:
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
        with open(self.stats_path, "r") as f:
            data = json.load(f)
        return DeepDreamStats(
            step=data["step"],
            activation=data["activation"],
            penalties=data["penalties"],
            total_loss=data["total_loss"],
        )

    def get_final_image(self) -> Image.Image:
        return Image.open(self.final_image_path)

    def get_final_latent(self, device="cpu") -> torch.Tensor:
        with safe_open(self.final_latent_path, framework="pt", device=device) as f:
            key = list(f.keys())[0]
            return f.get_tensor(key)


# Changed: The leaf value is now a List of results instead of a single object
TimestepDict = Dict[int, Dict[NoiseMode, List[DeepDreamResult]]]
ChannelDict = Dict[int, TimestepDict]


def _parse_timestep_dir(timestep_path: Path) -> List[DeepDreamResult]:
    images_dir = timestep_path / "images"
    latents_dir = timestep_path / "latents"
    intermediate_dir = timestep_path / "intermediate"
    stats_path = timestep_path / "stats.json"

    results: List[DeepDreamResult] = []

    if not images_dir.exists():
        return results

    # 1. Identify all variants by looking at the final images
    # Pattern: deep_dream_image_{idx:04d}.png
    final_image_files = sorted(images_dir.glob("deep_dream_image_*.png"))

    for img_path in final_image_files:
        try:
            # Extract index (e.g. "0000" from "deep_dream_image_0000.png")
            idx_str = img_path.stem.split("_")[-1]
            idx = int(idx_str)
        except ValueError:
            continue

        # 2. Find corresponding latent
        latent_path = latents_dir / f"deep_dream_latent_{idx_str}.safetensors"

        # 3. Find specific intermediate steps for this variant index
        variant_steps: List[IntermediateStep] = []
        if intermediate_dir.exists():
            # Iterate through step directories (step_0000, step_0010, etc.)
            for step_dir in sorted(intermediate_dir.glob("step_*")):
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
                except ValueError:
                    continue

        # Sort steps just in case
        variant_steps.sort(key=lambda x: x.step_idx)

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

    # Sort results by variant index (0, 1, 2...)
    return sorted(results, key=lambda x: x.variant_idx)


def _process_root_directory(
    root_path: Path, noise_mode: NoiseMode, raw_results: ChannelDict, sae_results: ChannelDict
) -> None:
    if not root_path or not root_path.exists():
        return

    for rank_dir in root_path.glob("fabric_rank_*"):
        # Process Standard Deep Dream
        deep_dream_dir = rank_dir / "deep_dream"
        if deep_dream_dir.exists():
            for channel_dir in deep_dream_dir.glob("channel_*"):
                try:
                    channel_id = int(channel_dir.name.split("_")[-1])
                except ValueError:
                    continue

                if channel_id not in raw_results:
                    raw_results[channel_id] = {}

                for timestep_dir in channel_dir.glob("timestep_*"):
                    try:
                        timestep = int(timestep_dir.name.split("_")[-1])
                    except ValueError:
                        continue

                    if timestep not in raw_results[channel_id]:
                        raw_results[channel_id][timestep] = {}

                    raw_results[channel_id][timestep][noise_mode] = _parse_timestep_dir(
                        timestep_dir
                    )

        # Process SAE Deep Dream
        deep_dream_sae_dir = rank_dir / "deep_dream_sae"
        if deep_dream_sae_dir.exists():
            for channel_dir in deep_dream_sae_dir.glob("channel_*"):
                try:
                    channel_id = int(channel_dir.name.split("_")[-1])
                except ValueError:
                    continue

                if channel_id not in sae_results:
                    sae_results[channel_id] = {}

                for timestep_dir in channel_dir.glob("timestep_*"):
                    try:
                        timestep = int(timestep_dir.name.split("_")[-1])
                    except ValueError:
                        continue

                    if timestep not in sae_results[channel_id]:
                        sae_results[channel_id][timestep] = {}

                    sae_results[channel_id][timestep][noise_mode] = _parse_timestep_dir(
                        timestep_dir
                    )


def get_deep_dream_results(noise_dir: Path, no_noise_dir: Path) -> tuple[ChannelDict, ChannelDict]:
    raw_channels: ChannelDict = {}
    sae_channels: ChannelDict = {}

    _process_root_directory(noise_dir, "noise", raw_channels, sae_channels)
    _process_root_directory(no_noise_dir, "no_noise", raw_channels, sae_channels)

    return raw_channels, sae_channels


if __name__ == "__main__":
    dd_noise_path = Path(
        "/net/pr2/projects/plgrid/plggailpwmm/aszokalski/diffusion-deep-dream-research/outputs/dd_sweep_gradient_smoothing/Stage.deep_dream/multirun/2026-01-18/21-31-48/1"
    )
    dd_no_noise_path = dd_noise_path

    raw_dd, sae_dd = get_deep_dream_results(dd_noise_path, dd_no_noise_path)

    print(f"Raw Channels Loaded: {len(raw_dd)}")
    print(f"SAE Channels Loaded: {len(sae_dd)}")

    print(raw_dd[19])
