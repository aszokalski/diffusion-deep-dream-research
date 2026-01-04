from typing import cast

import numpy as np
import torch
from loguru import logger
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, TimestepAnalysisStageConfig, Stage, \
    CaptureStageConfig
from diffusion_deep_dream_research.utils.capture_results_reading_utils import get_batches
import json
import pickle

def run_timestep_analysis(config: ExperimentConfig):
    stage_config = cast(TimestepAnalysisStageConfig, config.stage_config)
    capture_config = cast(CaptureStageConfig, config.stages[Stage.capture])
    use_sae = config.use_sae

    capture_results_abs_path = config.project_root / stage_config.capture_results_dir
    logger.info(f"Using capture results from \n [relative]: {stage_config.capture_results_dir} \n [absolute]: {capture_results_abs_path}")

    batches = get_batches(capture_results_abs_path)
    total_batch_size = capture_config.batch_size * capture_config.num_images_per_prompt
    n_batches = len(batches)
    total_size = total_batch_size * n_batches
    logger.info(f"Found {n_batches} batches with total size {total_size} (total_batch_size={total_batch_size})")

    first_batch = batches[0]
    sorted_timesteps = sorted(first_batch.activations_per_timestep.keys())
    timestep_to_idx = {ts: i for i, ts in enumerate(sorted_timesteps)}
    first_act = first_batch.activations_per_timestep[sorted_timesteps[0]].raw
    n_channels = first_act.shape[-1]
    n_timesteps = len(sorted_timesteps)
    logger.info(f"Found {n_channels} channels and {n_timesteps} timesteps (from data)")

    count_in_top_k_activations = np.zeros((n_channels, n_timesteps), dtype=np.float32)

    for batch in batches:
        for timestep, activations in batch.activations_per_timestep.items():
            raw_activations = activations.raw  # (total_batch_size, n_channels)

            if timestep not in timestep_to_idx:
                logger.warning(f"Skipping unexpected timestep {timestep}")
                continue
            t_idx = timestep_to_idx[timestep]

            act_tensor = torch.from_numpy(raw_activations)
            top_k_indices = torch.topk(act_tensor, k=stage_config.top_k, dim=-1).indices  # (total_batch_size, k)

            counts_tensor = torch.bincount(top_k_indices.flatten(), minlength=n_channels)
            count_in_top_k_activations[:, t_idx] += counts_tensor.cpu().numpy()


    frequency_in_top_k = count_in_top_k_activations / total_size #(n_channels, n_timesteps)

    x_observed = np.array(sorted_timesteps)
    x_full = np.arange(stage_config.total_timesteps+1)

    active_timesteps = [] # (channel,)
    activity_peaks = [] # (channel,)

    for channel_idx in range(n_channels):
        y_observed = frequency_in_top_k[channel_idx]

        if np.sum(y_observed) == 0:
            active_timesteps.append([])
            activity_peaks.append([])
            continue

        # Interpolation to stretch to actual timesteps
        f_interp = interp1d(x_observed, y_observed, kind='linear', bounds_error=False, fill_value=0)
        y_full = f_interp(x_full)

        # Active timesteps
        active_mask = y_full > 0
        channel_active_steps = x_full[active_mask].tolist()
        active_timesteps.append(channel_active_steps)

        # Activity peaks
        peaks, properties = find_peaks(
            y_full,
            height=stage_config.peak_threshold,
            distance=stage_config.peak_separation
        )

        current_peaks = []
        for p_idx, p_height in zip(peaks, properties['peak_heights']):
            current_peaks.append((int(x_full[p_idx]), float(p_height)))

        # Edges (start and end)
        if y_full[0] > 0 and len(y_full) > 1 and y_full[0] > y_full[1]:
            current_peaks.append((int(x_full[0]), float(y_full[0])))
        if y_full[-1] > 0 and len(y_full) > 1 and y_full[-1] > y_full[-2]:
            current_peaks.append((int(x_full[-1]), float(y_full[-1])))

        # Only keep the top
        current_peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = current_peaks[:stage_config.top_peak_count]
        top_peaks_no_height = [tp[0] for tp in top_peaks]
        activity_peaks.append(top_peaks_no_height)

    # Save results
    with open("timestep_analysis.json", "w") as f:
        analysis_dict = {"active_timesteps": active_timesteps, "activity_peaks": activity_peaks}
        json.dump(analysis_dict, f)

    with open("frequency_in_top_k_and_sorted_timesteps.pkl", "wb") as f:
        pickle.dump((frequency_in_top_k, sorted_timesteps), f)

    logger.info(f"Analysis complete.")