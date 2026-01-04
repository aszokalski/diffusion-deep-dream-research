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
    pkl_filename = f"frequency_in_top_k_sorted_timesteps_max_activation{suffix}.pkl"
    active_timesteps_json_filename = f"active_timesteps{suffix}.json"
    activity_peaks_json_filename = f"activity_peaks{suffix}.json"
    dataset_examples_json_filename = f"dataset_examples{suffix}.json"

    logger.info(f"Loading analysis data from {timesteps_analysis_results_abs_path}...")

    with open(timesteps_analysis_results_abs_path / pkl_filename, "rb") as f:
        frequency_in_top_k, sorted_timesteps, max_activation = pickle.load(f)

    with open(timesteps_analysis_results_abs_path / activity_peaks_json_filename, "r") as f:
        activity_peaks = json.load(f)

    with open(timesteps_analysis_results_abs_path / dataset_examples_json_filename, "r") as f:
        dataset_examples = json.load(f)

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

    active_channels = []
    for channel_idx in tqdm(range(n_channels), desc="Plotting", mininterval=10.0, ascii=True, ncols=80):
        channel_name = f"channel_{channel_idx:04d}"

        if np.sum(frequency_in_top_k[channel_idx]) > 0:
            # Note the active channels
            active_channels.append(channel_name)

        y_observed = frequency_in_top_k[channel_idx]

        f_interp = interp1d(x_observed, y_observed, kind='linear', bounds_error=False, fill_value=0)
        y_full = f_interp(x_full)

        # two plots
        fig, (ax_up, ax_down) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

        # Upper Axes: Frequency in Top-K, Active Timesteps Area Under Curve (above 0)
        # Plot 1: Raw Data Points
        ax_up.plot(x_observed, y_observed, 'o', label='Observed', alpha=0.4, color='gray')

        # Plot 2: Interpolated Curve
        ax_up.plot(x_full, y_full, '-', label='Interpolated', alpha=0.9, color='#2ecc71', linewidth=2)

        # Plot 3: Active Timesteps Area Under Curve (above 0)
        ax_up.fill_between(x_full, 0, y_full, alpha=0.15, color='#2ecc71', label='Active Timesteps')

        # Plot 4: Peaks
        channel_peaks = activity_peaks[channel_idx]
        for t_peak in channel_peaks:
            if 0 <= t_peak < len(y_full):
                val_peak = y_full[t_peak]
                ax_up.plot(t_peak, val_peak, 'x', color='#e74c3c', markersize=10, markeredgewidth=3, label='Peak')
                ax_up.text(t_peak, val_peak, f" T={t_peak}", verticalalignment='bottom', fontsize=9, fontweight='bold')

        ax_up.set_xlabel("Timestep")
        ax_up.set_ylabel("Frequency in Top-K")
        ax_up.set_title(f"Channel {channel_idx} Activity Profile{' (SAE)' if sae else ''}")

        # Bottom Axes: Max Activation
        max_activation_at_timestep = max_activation[channel_idx]
        ax_down.plot(sorted_timesteps, max_activation_at_timestep, '-', color='#3498db', label='Max Activation')

        ax_down.set_xlabel("Timestep")
        ax_down.set_ylabel("Max Activation")
        ax_down.set_title(f"Channel {channel_idx} Max Activation")

        # Styling
        for ax in (ax_up, ax_down):
            ax.axhline(0, color='black', linewidth=1, alpha=0.3)
            ax.invert_xaxis()  # Diffusion goes 1000 -> 0
            ax.legend()
            ax.grid(True, alpha=0.2)

        save_path = output_dir / f"{channel_name}.png"
        fig.savefig(save_path, dpi=100)

        plt.close(fig)

    with open(output_dir / f"active_channels{suffix}.json", "w") as f:
        json.dump(active_channels, f, indent=4)

    logger.info("Plot generation complete.")


def run_plots(config: ExperimentConfig):
    stage_config = cast(PlotsStageConfig, config.stage_config)
    timestep_analysis_config = cast(TimestepAnalysisStageConfig, config.stages[Stage.timestep_analysis])
    use_sae = config.use_sae

    timesteps_analysis_results_abs_path = config.outputs_dir / stage_config.timestep_analysis_results_dir

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