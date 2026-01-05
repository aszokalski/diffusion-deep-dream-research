import random
import textwrap
from pathlib import Path
from typing import cast, Optional, Tuple

import imageio
import numpy as np
import pickle
import json
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from tqdm import tqdm
from loguru import logger
from PIL import Image

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, PlotsStageConfig, \
    TimestepAnalysisStageConfig, Stage


def get_random_example(
        dataset_examples: dict,
        channel_idx: int,
        timestep: int,
        project_root: Path
) -> Tuple[Optional[Image.Image], str]:
    """
    Retrieves a random (image, prompt) pair for the given channel and timestep.
    """
    # Keys in JSON are always strings
    c_key = str(channel_idx)
    t_key = str(timestep)

    if c_key not in dataset_examples:
        return None, "No examples for this channel"

    if t_key not in dataset_examples[c_key]:
        # Fallback: try to find the closest timestep if exact match missing
        # (Optional robustness, strictly speaking dataset should cover it)
        return None, f"No examples for timestep {timestep}"

    examples_list = dataset_examples[c_key][t_key]
    if not examples_list:
        return None, "Empty example list"

    # Pick a random example
    prompt, rel_path_str = random.choice(examples_list)

    # Construct full path. rel_path_str is relative to project root
    # e.g. "outputs/experiment/..."
    image_path = project_root / rel_path_str

    if not image_path.exists():
        return None, f"Image not found: {image_path.name}"

    try:
        img = Image.open(image_path).convert("RGB")
        return img, prompt
    except Exception as e:
        logger.warning(f"Failed to open image {image_path}: {e}")
        return None, f"Error loading image"


def generate_plots(
        *,
        stage_config: PlotsStageConfig,
        timestep_analysis_config: TimestepAnalysisStageConfig,
        timesteps_analysis_results_abs_path: Path,
        project_root: Path,  # Added to resolve image paths
        sae: bool
) -> None:
    suffix = "_sae" if sae else ""
    pkl_filename = f"frequency_in_top_k_sorted_timesteps_max_activation{suffix}.pkl"
    # Note: JSON filenames usually don't need suffix if they were saved differently,
    # but strictly following your naming convention:
    active_timesteps_json_filename = f"active_timesteps{suffix}.json"
    activity_peaks_json_filename = f"activity_peaks{suffix}.json"
    dataset_examples_json_filename = f"dataset_examples{suffix}.json"

    logger.info(f"Loading analysis data from {timesteps_analysis_results_abs_path}...")

    # Load Data
    with open(timesteps_analysis_results_abs_path / pkl_filename, "rb") as f:
        frequency_in_top_k, sorted_timesteps, max_activation = pickle.load(f)

    with open(timesteps_analysis_results_abs_path / activity_peaks_json_filename, "r") as f:
        activity_peaks = json.load(f)  # list of lists (for each channel)

    with open(timesteps_analysis_results_abs_path / dataset_examples_json_filename, "r") as f:
        dataset_examples = json.load(f)  # dict: str(channel) -> str(timestep) -> list

    output_dir = Path("plots_sae" if sae else "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving GIFs to {output_dir}")

    # Interpolation setup
    x_observed = np.array(sorted_timesteps)
    x_full = np.arange(0, timestep_analysis_config.total_timesteps + 1)
    n_channels = frequency_in_top_k.shape[0]

    active_channels = []

    # Iterate over channels
    for channel_idx in tqdm(range(n_channels), desc="Generating GIFs", mininterval=10.0, ascii=True, ncols=80):
        channel_name = f"channel_{channel_idx:04d}"

        # Skip if channel has no activity
        if np.sum(frequency_in_top_k[channel_idx]) == 0:
            continue

        active_channels.append(channel_name)

        y_observed = frequency_in_top_k[channel_idx]

        # Interpolate for the plots
        f_interp = interp1d(x_observed, y_observed, kind='linear', bounds_error=False, fill_value=0)
        y_full = f_interp(x_full)

        channel_peaks = activity_peaks[channel_idx]
        # If no peaks detected but channel is active (rare), fallback to just middle frame or skip
        if not channel_peaks:
            # Create a dummy peak at max value just to show one static frame
            t_max = int(np.argmax(y_full))
            iter_peaks = [t_max]
            is_gif = False
        else:
            iter_peaks = channel_peaks
            is_gif = True

        frames = []

        # Generate one frame per peak
        for t_target_peak in iter_peaks:
            # --- Layout Setup ---
            # 2 Columns: Left (Plots), Right (Image & Prompt)
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1],
                          left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)

            ax_up = fig.add_subplot(gs[0, 0])  # Frequency Plot
            ax_down = fig.add_subplot(gs[1, 0])  # Max Activation Plot
            ax_img = fig.add_subplot(gs[:, 1])  # Image (takes full height of right col)

            # --- 1. Top Left: Frequency Plot ---
            ax_up.plot(x_observed, y_observed, 'o', label='Observed', alpha=0.4, color='gray')
            ax_up.plot(x_full, y_full, '-', label='Interpolated', alpha=0.9, color='#2ecc71', linewidth=2)
            ax_up.fill_between(x_full, 0, y_full, alpha=0.15, color='#2ecc71')

            # Draw all peaks, but highlight the TARGET one
            for t_peak in channel_peaks:
                if 0 <= t_peak < len(y_full):
                    val_peak = y_full[t_peak]

                    ax_up.annotate(
                        f"T={t_peak}",
                        xy=(t_peak, val_peak),  # The point to label
                        xytext=(0, -10),  # OFFSET: 0 points horizontal, -10 points vertical (down)
                        textcoords='offset points',  # Interpret xytext as screen points, not data values
                        ha='center',  # Horizontal Alignment: Center
                        va='top',  # Vertical Alignment: Top of text matches the offset point
                        fontsize=10,
                        fontweight='bold',
                        color='black'
                    )
                    ax_up.plot(t_peak, val_peak, 'x', markersize=10, markeredgewidth=3, color='black', zorder=11)

                    if t_peak == t_target_peak:
                        ax_up.plot(t_peak, val_peak, 'o', color='#e74c3c', markersize=15,
                                   zorder=10, alpha=0.6)


            ax_up.set_title(f"Channel {channel_idx} Activity Profile", fontsize=14)
            ax_up.set_ylabel("Freq in Top-K")
            ax_up.axhline(0, color='black', linewidth=1, alpha=0.3)
            ax_up.invert_xaxis()
            ax_up.grid(True, alpha=0.2)
            # ax_up.legend(loc='upper left')

            # --- 2. Bottom Left: Max Activation ---
            max_activation_at_timestep = max_activation[channel_idx]
            ax_down.plot(sorted_timesteps, max_activation_at_timestep, '-', color='#3498db', linewidth=2)

            # Highlight current timestep on this plot too
            if t_target_peak in x_full:
                # Find closest actual timestep index for plotting the dot
                # Since max_activation is defined on 'sorted_timesteps', we interpret t_target_peak linearly
                # Simplification: just plot a vertical line
                ax_down.axvline(t_target_peak, color='#e74c3c', linestyle='--', alpha=0.8, label=f'T={t_target_peak}')

            ax_down.set_title(f"Channel {channel_idx} Max Activation", fontsize=14)
            ax_down.set_xlabel("Timestep")
            ax_down.set_ylabel("Activation Value")
            ax_down.invert_xaxis()
            ax_down.grid(True, alpha=0.2)

            # --- 3. Right: Image & Prompt ---
            img, prompt_text = get_random_example(dataset_examples, channel_idx, t_target_peak, project_root)

            if img:
                ax_img.imshow(img)
                ax_img.axis('off')
            else:
                # Placeholder if missing
                ax_img.text(0.5, 0.5, "Image Not Available", ha='center', va='center', fontsize=16)
                ax_img.axis('off')

            # Add Prompt Text at the bottom of the image axes
            wrapped_prompt = "\n".join(textwrap.wrap(f"{prompt_text}", width=80))
            ax_img.set_title(wrapped_prompt, fontsize=12, y=-0.15, loc='center', wrap=True)

            # --- Save Frame ---
            # Render the figure to a numpy array buffer
            fig.canvas.draw()
            image_from_plot = np.array(fig.canvas.renderer.buffer_rgba())

            # buffer_rgba gives RGBA, but we likely want RGB for the GIF
            # Convert RGBA -> RGB to be safe
            if image_from_plot.shape[2] == 4:
                image_from_plot = image_from_plot[:, :, :3]

            frames.append(image_from_plot)

            plt.close(fig)  # Clear memory

        # Save results
        if is_gif and len(frames) > 1:
            save_path = output_dir / f"{channel_name}.gif"
            # duration per frame in seconds. Loop=0 means infinite loop
            imageio.mimsave(save_path, frames, duration=1000 * stage_config.frame_duration, loop=0)
        elif len(frames) == 1:
            # Fallback to PNG if only one peak/frame
            save_path = output_dir / f"{channel_name}.png"
            imageio.imwrite(save_path, frames[0])

    # Save active channels list
    with open(output_dir / f"active_channels{suffix}.json", "w") as f:
        json.dump(active_channels, f, indent=4)

    logger.info("Plot/GIF generation complete.")


def run_plots(config: ExperimentConfig):
    stage_config = cast(PlotsStageConfig, config.stage_config)
    # Note: We need to access the TimestepAnalysis config to know 'total_timesteps'
    # Assuming standard config structure where stages are accessible
    timestep_analysis_config = cast(TimestepAnalysisStageConfig, config.stages[Stage.timestep_analysis])
    use_sae = config.use_sae

    timesteps_analysis_results_abs_path = config.outputs_dir / stage_config.timestep_analysis_results_dir

    logger.info(
        f"Using timestep analysis results from: {timesteps_analysis_results_abs_path}")

    generate_plots(
        stage_config=stage_config,
        timestep_analysis_config=timestep_analysis_config,
        timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
        project_root=config.project_root,  # Pass project root for resolving image paths
        sae=False
    )

    if use_sae:
        generate_plots(
            stage_config=stage_config,
            timestep_analysis_config=timestep_analysis_config,
            timesteps_analysis_results_abs_path=timesteps_analysis_results_abs_path,
            project_root=config.project_root,
            sae=True
        )