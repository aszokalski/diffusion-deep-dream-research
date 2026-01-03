from typing import cast
from loguru import logger

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, TimestepAnalysisStageConfig
from diffusion_deep_dream_research.utils.capture_results_reading_utils import get_batches


def run_timestep_analysis(config: ExperimentConfig):
    stage_config = cast(TimestepAnalysisStageConfig, config.stage_config)
    capture_results_abs_path = config.project_root / stage_config.capture_results_dir
    use_sae = config.use_sae

    logger.info(f"Using capture results from \n [relative]: {stage_config.capture_results_dir} \n [absolute]: {capture_results_abs_path}")

    batches = get_batches(capture_results_abs_path)

    for batch in batches:
        for timestep, activations in batch.activations_per_timestep.items():
            raw_activations = activations.raw
            if use_sae:
                sae_activations = activations.encoded









