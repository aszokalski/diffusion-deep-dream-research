from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
from typing import Dict, List, Literal, cast

from loguru import logger
import numpy as np
from scipy.interpolate import interp1d

from diffusion_deep_dream_research.config.config_schema import (
    ExperimentConfig,
    RepresentationStageConfig,
    Stage,
    TimestepAnalysisStageConfig,
)
from diffusion_deep_dream_research.utils.deep_dream_results_reading_utils import (
    DeepDreamResult,  # Importing for type hinting
    get_deep_dream_results,
)
from diffusion_deep_dream_research.utils.prior_results_reading_utils import (
    ChannelPriors,
    get_prior_results,
)


@dataclass
class IndexMetadata:
    active_channels: List[int]
    sorted_timesteps: List[int]
    analysis_type: str
    total_channels: int


@dataclass
class ChannelMeta:
    activity_profile: np.ndarray
    max_activation: np.ndarray
    peaks: List[int]


@dataclass
class TimestepData:
    dataset_examples: List[Dict[str, str]]
    # Map: NoiseMode -> List of DeepDreamResults (variants)
    deep_dream: Dict[Literal["noise", "no_noise"], List[DeepDreamResult]] = field(
        default_factory=dict
    )


@dataclass
class ChannelData:
    id: int
    meta: ChannelMeta
    priors: ChannelPriors
    timesteps: Dict[int, TimestepData]


def process_representation(
    config: ExperimentConfig,
    stage_config: RepresentationStageConfig,
    timestep_analysis_config: TimestepAnalysisStageConfig,
    use_sae: bool,
):
    logger.info(f"Starting process_representation | Use SAE: {use_sae}")

    suffix = "_sae" if use_sae else ""
    analysis_type = "sae" if use_sae else "non-sae"

    # Input Paths
    timesteps_results_path = config.outputs_dir / stage_config.timestep_analysis_results_dir
    prior_results_path = config.outputs_dir / stage_config.prior_results_dir
    dd_noise_path = Path(config.outputs_dir) / stage_config.deep_dream_results_dir_noise
    dd_no_noise_path = (
        Path(config.outputs_dir) / stage_config.deep_dream_results_dir_no_noise
        if stage_config.deep_dream_results_dir_no_noise
        else None
    )

    logger.debug(f"Timesteps Results Path: {timesteps_results_path}")
    logger.debug(f"Prior Results Path: {prior_results_path}")
    logger.debug(f"Deep Dream Noise Path: {dd_noise_path}")
    logger.debug(f"Deep Dream No-Noise Path: {dd_no_noise_path}")

    # Output Paths
    mode_dir = Path(analysis_type)
    channels_dir = mode_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running in mode: {analysis_type.upper()}")
    logger.info(f"Saving to: {mode_dir}")

    # Loading results
    logger.info(f"Loading Analysis from {timesteps_results_path}")

    freq_path = (
        timesteps_results_path / f"frequency_in_top_k_sorted_timesteps_max_activation{suffix}.pkl"
    )
    logger.debug(f"Reading frequency/activation data from {freq_path}")
    with open(freq_path, "rb") as f:
        frequency_matrix, sorted_timesteps, max_activation_matrix = pickle.load(f)
    logger.info(f"Loaded frequency_matrix shape: {frequency_matrix.shape}")
    logger.info(f"Loaded max_activation_matrix shape: {max_activation_matrix.shape}")
    logger.info(f"Loaded {len(sorted_timesteps)} sorted timesteps")

    peaks_path = timesteps_results_path / f"activity_peaks{suffix}.json"
    logger.debug(f"Reading activity peaks from {peaks_path}")
    with open(peaks_path, "r") as f:
        activity_peaks = json.load(f)  # channel -> list of timesteps
    logger.info(f"Loaded activity peaks for {len(activity_peaks)} channels")

    examples_path = timesteps_results_path / f"dataset_examples{suffix}.json"
    logger.debug(f"Reading dataset examples from {examples_path}")
    with open(examples_path, "r") as f:
        dataset_examples = json.load(f)  # channel -> timestep -> list of (prompt, path)
    logger.info(f"Loaded dataset examples for {len(dataset_examples)} channels")

    logger.info(f"Indexing Prior Results from {prior_results_path}")
    prior_results = get_prior_results(prior_results_path)
    # Select specific priors map (raw or sae)
    priors_map = prior_results.sae if use_sae else prior_results.raw
    logger.info(f"Loaded {len(priors_map)} prior entries (SAE mode: {use_sae})")

    logger.debug(f"Indexing Deep Dream Results from {dd_noise_path} and {dd_no_noise_path}")
    raw_dd, sae_dd = get_deep_dream_results(dd_noise_path, dd_no_noise_path)
    # Select specific deep dream map (raw or sae)
    deep_dream_map = sae_dd if use_sae else raw_dd
    logger.info(f"Loaded {len(deep_dream_map)} deep dream entries (SAE mode: {use_sae})")

    index_metadata = IndexMetadata(
        active_channels=[],
        sorted_timesteps=sorted_timesteps,
        analysis_type=analysis_type,
        total_channels=frequency_matrix.shape[0],
    )

    x_observed = np.array(sorted_timesteps)
    x_full = np.arange(0, timestep_analysis_config.total_timesteps)
    n_channels = frequency_matrix.shape[0]

    logger.info(f"Processing {n_channels} channels...")

    for channel_id in range(n_channels):
        if channel_id % 100 == 0:
            logger.info(f"Processing channel batch starting at {channel_id}/{n_channels}")

        if sum(frequency_matrix[channel_id]) > 0:
            index_metadata.active_channels.append(channel_id)

        # 1. Prepare Channel Metadata (Interpolation)
        # frequency in top-k[timestep]
        y_observed = frequency_matrix[channel_id]
        f_interp = interp1d(
            x_observed, y_observed, kind="linear", bounds_error=False, fill_value=0
        )
        y_full = f_interp(x_full)

        # max-activation[timestep]
        act_observed = max_activation_matrix[channel_id]
        f_act_interp = interp1d(
            x_observed, act_observed, kind="linear", bounds_error=False, fill_value=0
        )
        act_full = f_act_interp(x_full)

        channel_meta = ChannelMeta(
            activity_profile=y_full,
            max_activation=act_full,
            peaks=activity_peaks[channel_id],
        )

        # 2. Get Priors Object
        # Default to empty ChannelPriors if not found for this channel
        channel_priors = priors_map.get(channel_id, ChannelPriors(images_with_latents=[]))

        # 3. Build Timestep Map
        timesteps_map: Dict[int, TimestepData] = {}

        # Collect relevant timesteps (union of dataset examples and deep dream results)
        relevant_timesteps = set()
        c_key = str(channel_id)
        if c_key in dataset_examples:
            relevant_timesteps.update(int(k) for k in dataset_examples[c_key].keys())
        if channel_id in deep_dream_map:
            relevant_timesteps.update(deep_dream_map[channel_id].keys())

        logger.debug(f"Channel {channel_id} has {len(relevant_timesteps)} relevant timesteps")

        for t in relevant_timesteps:
            timestep_obj = TimestepData(dataset_examples=[], deep_dream={})

            # Add Dataset Examples
            if c_key in dataset_examples and str(t) in dataset_examples[c_key]:
                channel_timestep_examples = dataset_examples[c_key][str(t)]
                timestep_obj.dataset_examples = [
                    {"prompt": ex[0], "path": str(ex[1])} for ex in channel_timestep_examples
                ]

            # Add Deep Dreams
            if channel_id in deep_dream_map and t in deep_dream_map[channel_id]:
                # deep_dream_map[channel][t] is Dict[NoiseMode, List[DeepDreamResult]]
                result_modes = deep_dream_map[channel_id][t]

                # Directly assign the dictionary of lists of result objects
                # No manual dictionary construction needed; we keep the objects.
                timestep_obj.deep_dream = result_modes

            timesteps_map[t] = timestep_obj

        # 4. Construct Final Channel Data Object
        channel_data = ChannelData(
            id=channel_id,
            meta=channel_meta,
            priors=channel_priors,
            timesteps=timesteps_map,
        )

        # 5. Save Shard
        shard_path = channels_dir / f"channel_{channel_id:05d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(channel_data, f)

    logger.info(f"Finished processing all {n_channels} channels.")
    logger.info(f"Total active channels identified: {len(index_metadata.active_channels)}")

    # --- 6. SAVE GLOBAL METADATA ---
    meta_path = mode_dir / "index_metadata.pkl"
    logger.info(f"Saving Metadata to {meta_path}")
    with open(meta_path, "wb") as f:
        pickle.dump(index_metadata, f)
    logger.info(f"Metadata saved successfully. Process for {analysis_type} complete.")


def run_representation(config: ExperimentConfig):
    logger.info("Initializing run_representation stage")
    stage_config = cast(RepresentationStageConfig, config.stage_config)
    timestep_analysis_config = cast(
        TimestepAnalysisStageConfig, config.stages[Stage.timestep_analysis]
    )

    # Process Standard (Raw) Channels
    logger.info("Initiating Standard (Raw) Channel Processing")
    process_representation(config, stage_config, timestep_analysis_config, False)

    # Process SAE Channels (if enabled)
    if config.use_sae:
        logger.info("Initiating SAE Channel Processing (config.use_sae=True)")
        process_representation(config, stage_config, timestep_analysis_config, True)
    else:
        logger.info("Skipping SAE Channel Processing (config.use_sae=False)")

    logger.info("run_representation stage finished.")
