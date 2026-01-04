from typing import Callable

import hydra
import submitit
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, register_configs, Stage
from diffusion_deep_dream_research.stages.s00_provision import run_provision
from loguru import logger

from diffusion_deep_dream_research.stages.s01_capture import run_capture
from diffusion_deep_dream_research.stages.s02_timestep_analysis import run_timestep_analysis
from diffusion_deep_dream_research.stages.s03_plots import run_plots
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging

register_configs()

stages: dict[Stage, Callable[[ExperimentConfig], None]] = {
    Stage.provision: run_provision,
    Stage.capture: run_capture,
    Stage.timestep_analysis: run_timestep_analysis,
    Stage.plots: run_plots
}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    try:
        env = submitit.JobEnvironment()
    except RuntimeError as e:
        logger.warning("Submitit env not found. If you want to run on Submitit make sure to use the --multirun flag.")
        env = None

    if env is not None:
        setup_distributed_logging(env.global_rank)
        logger.info(f"Submitit Job Environment: {env}")

    config: ExperimentConfig = OmegaConf.to_object(cfg)
    logger.info(f"Running with config: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Executing stage: {config.stage}...")
    stages[config.stage](
        config
    )
    logger.info("Done!")




if __name__ == "__main__":
    main()