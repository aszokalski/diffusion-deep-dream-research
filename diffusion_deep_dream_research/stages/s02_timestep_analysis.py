import json
import pickle
from typing import cast

from loguru import logger
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import torch
from tqdm import tqdm

from diffusion_deep_dream_research.config.config_schema import (
    CaptureStageConfig,
    ExperimentConfig,
    Stage,
    TimestepAnalysisStageConfig,
)
from diffusion_deep_dream_research.utils.capture_results_reading_utils import Batch, get_batches


def analysis(
    *,
    batches: list[Batch],
    capture_config: CaptureStageConfig,
    stage_config: TimestepAnalysisStageConfig,
    sae: bool,
) -> None:
    logger.info(f"Starting analysis (SAE={sae})...")

    total_batch_size = capture_config.batch_size * capture_config.num_images_per_prompt
    n_batches = len(batches)
    total_size = total_batch_size * n_batches  # Number of total generations (prompts repeated)
    logger.info(
        f"Found {n_batches} batches with total size {total_size} (total_batch_size={total_batch_size})"
    )

    first_batch = batches[0]
    sorted_timesteps = sorted(first_batch.activations_per_timestep.keys())

    timestep_to_idx = {ts: i for i, ts in enumerate(sorted_timesteps)}

    if sae:
        first_act = first_batch.activations_per_timestep[sorted_timesteps[0]].encoded
    else:
        first_act = first_batch.activations_per_timestep[sorted_timesteps[0]].raw

    assert isinstance(first_act, np.ndarray)
    n_channels = first_act.shape[-1]
    n_timesteps = len(sorted_timesteps)

    logger.info(f"Found {n_channels} channels and {n_timesteps} timesteps (from data)")

    count_in_top_k_activations = np.zeros((n_channels, n_timesteps), dtype=np.float32)
    max_activation = np.zeros((n_channels, n_timesteps), dtype=np.float32)
    dataset_examples: dict[int, dict[int, list[tuple[str, str]]]] = {
        channel_idx: {timestep: [] for timestep in sorted_timesteps}
        for channel_idx in range(n_channels)
    }  # channel -> timestep(full) -> list of (prompt, generated_image_path)

    for batch in tqdm(batches, mininterval=10.0, ascii=True, ncols=80):
        prompts = batch.prompts
        generated_image_paths = batch.generated_image_paths

        for timestep, activations in batch.activations_per_timestep.items():
            if sae:
                np_activations = activations.encoded  # (total_batch_size, n_channels)
            else:
                np_activations = activations.raw  # (total_batch_size, n_channels)

            if timestep not in timestep_to_idx:
                logger.warning(f"Skipping unexpected timestep {timestep}")
                continue
            t_idx = timestep_to_idx[timestep]

            # Calculate frequency of activations in top k
            act_tensor = torch.from_numpy(np_activations)
            top_k_indices = torch.topk(
                act_tensor, k=stage_config.top_k, dim=-1
            ).indices  # (total_batch_size, k)

            counts_tensor = torch.bincount(top_k_indices.flatten(), minlength=n_channels)
            count_in_top_k_activations[:, t_idx] += counts_tensor.cpu().numpy()

            # Update max activations for every channel at every timestep
            max_activation[:, t_idx] = np.maximum(
                max_activation[:, t_idx], np_activations.max(axis=0)
            )

            # Assign dataset examples
            for batch_idx, (prompt, generated_image_path) in enumerate(
                zip(prompts, generated_image_paths)
            ):
                top_k_indices_for_batch_idx = top_k_indices[batch_idx]
                for channel_idx in top_k_indices_for_batch_idx:
                    dataset_examples[channel_idx.detach().cpu().item()][timestep].append(
                        (prompt, str(generated_image_path))
                    )

    frequency_in_top_k = count_in_top_k_activations / total_size  # (n_channels, n_timesteps)

    x_observed = np.array(sorted_timesteps)
    x_full = np.arange(stage_config.total_timesteps + 1)

    active_timesteps = []  # (channel,)
    activity_peaks = []  # (channel,)

    for channel_idx in range(n_channels):
        y_observed = frequency_in_top_k[channel_idx]

        if np.sum(y_observed) == 0:
            active_timesteps.append([])
            activity_peaks.append([])
            continue

        # Interpolation to stretch to actual timesteps
        f_interp = interp1d(
            x_observed, y_observed, kind="linear", bounds_error=False, fill_value=0
        )
        y_full = f_interp(x_full)

        # Active timesteps
        active_mask = y_full > 0
        channel_active_steps = x_full[active_mask].tolist()
        active_timesteps.append(channel_active_steps)

        # Activity peaks
        peaks, properties = find_peaks(
            y_full,
            height=stage_config.peak_threshold,
            distance=stage_config.peak_separation,
            plateau_size=1,  # if there are flat peaks
        )

        current_peaks = []
        for p_idx, p_height in zip(peaks, properties["peak_heights"]):
            current_peaks.append((int(x_full[p_idx]), float(p_height)))

        # Only keep the top
        current_peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = current_peaks[: stage_config.top_peak_count]
        top_peaks_no_height = [tp[0] for tp in top_peaks]
        activity_peaks.append(top_peaks_no_height)

    # Save results
    with open(f"active_timesteps{'_sae' if sae else ''}.json", "w") as f:
        json.dump(active_timesteps, f)

    with open(f"activity_peaks{'_sae' if sae else ''}.json", "w") as f:
        json.dump(activity_peaks, f)

    with open(f"dataset_examples{'_sae' if sae else ''}.json", "w") as f:
        json.dump(dataset_examples, f)

    with open(
        f"frequency_in_top_k_sorted_timesteps_max_activation{'_sae' if sae else ''}.pkl", "wb"
    ) as f:
        pickle.dump((frequency_in_top_k, sorted_timesteps, max_activation), f)

    logger.info(f"Analysis complete. (SAE={sae})")


def run_timestep_analysis(config: ExperimentConfig):
    stage_config = cast(TimestepAnalysisStageConfig, config.stage_config)
    capture_config = cast(CaptureStageConfig, config.stages[Stage.capture])
    use_sae = config.use_sae

    capture_results_abs_path = config.outputs_dir / stage_config.capture_results_dir
    logger.info(
        f"Using capture results from \n [relative]: {stage_config.capture_results_dir} \n [absolute]: {capture_results_abs_path}"
    )

    batches = get_batches(capture_results_abs_path)

    analysis(batches=batches, capture_config=capture_config, stage_config=stage_config, sae=False)

    if use_sae:
        analysis(
            batches=batches, capture_config=capture_config, stage_config=stage_config, sae=True
        )
