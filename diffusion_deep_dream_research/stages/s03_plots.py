from pathlib import Path
from typing import cast
import numpy as np
import pickle
import json
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from loguru import logger

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, PlotsStageConfig, \
    TimestepAnalysisStageConfig, Stage


def generate_plots(
        *,
        stage_config: PlotsStageConfig,
        timestep_analysis_config: TimestepAnalysisStageConfig,
        timesteps_analysis_results_abs_path: Path,
        sae: bool
) -> None:
    suffix = "_sae" if sae else ""
    pkl_filename = f"frequency_in_top_k_and_sorted_timesteps{suffix}.pkl"
    json_filename = f"timestep_analysis{suffix}.json"

    logger.info(f"Loading analysis data from {timesteps_analysis_results_abs_path}...")

    with open(timesteps_analysis_results_abs_path / pkl_filename, "rb") as f:
        frequency_in_top_k, sorted_timesteps = pickle.load(f)

    with open(timesteps_analysis_results_abs_path / json_filename, "r") as f:
        analysis_dict = json.load(f)
        activity_peaks = analysis_dict["activity_peaks"]

    output_dir = Path("plots_sae" if sae else "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    # Interpolation
    # repeated because it's not computationally
    # heavy but saving it would waste a lot of
    # space due to sparsity of the data
    x_observed = np.array(sorted_timesteps)
    x_full = np.arange(0, timestep_analysis_config.total_timesteps + 1)

    n_channels = frequency_in_top_k.shape[0]

    logger.info(f"Generating plots for {n_channels} channels...")

    for channel_idx in tqdm(range(n_channels), desc="Plotting", mininterval=10.0, ascii=True, ncols=80):
        if np.sum(frequency_in_top_k[channel_idx]) == 0:
            # Skip inactive channels
            continue

        y_observed = frequency_in_top_k[channel_idx]

        f_interp = interp1d(x_observed, y_observed, kind='linear', bounds_error=False, fill_value=0)
        y_full = f_interp(x_full)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot 1: Raw Data Points
        ax.plot(x_observed, y_observed, 'o', label='Observed', alpha=0.4, color='gray')

        # Plot 2: Interpolated Curve
        ax.plot(x_full, y_full, '-', label='Interpolated', alpha=0.9, color='#2ecc71', linewidth=2)

        # Plot 3: Active Timesteps Area Under Curve (above 0)
        ax.fill_between(x_full, 0, y_full, alpha=0.15, color='#2ecc71')

        # Plot 4: Peaks
        channel_peaks = activity_peaks[channel_idx]
        for t_peak in channel_peaks:
            if 0 <= t_peak < len(y_full):
                val_peak = y_full[t_peak]
                ax.plot(t_peak, val_peak, 'x', color='#e74c3c', markersize=10, markeredgewidth=3)
                ax.text(t_peak, val_peak, f" T={t_peak}", verticalalignment='bottom', fontsize=9, fontweight='bold')

        # Styling
        ax.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax.invert_xaxis()  # Diffusion goes 1000 -> 0
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Frequency in Top-K")
        ax.set_title(f"Channel {channel_idx} Activity Profile{' (SAE)' if sae else ''}")
        ax.legend()
        ax.grid(True, alpha=0.2)

        save_path = output_dir / f"channel_{channel_idx:04d}.png"
        fig.savefig(save_path, dpi=100)

        plt.close(fig)

    logger.info("Plot generation complete.")


def run_plots(config: ExperimentConfig):
    stage_config = cast(PlotsStageConfig, config.stage_config)
    timestep_analysis_config = cast(TimestepAnalysisStageConfig, config.stages[Stage.timestep_analysis])
    use_sae = config.use_sae

    timesteps_analysis_results_abs_path = config.project_root / stage_config.timestep_analysis_results_dir

    logger.info(
        f"Using timestep analysis results from \n [relative]: {stage_config.timestep_analysis_results_dir} \n [absolute]: {timesteps_analysis_results_abs_path}")

    generate_plots(
        stage_config=stage_config,
        timestep_analysis_config=timestep_analysis_config,
        timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
        sae=False
    )

    if use_sae:
        generate_plots(
            stage_config=stage_config,
            timestep_analysis_config=timestep_analysis_config,
            timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
            sae=True
        )