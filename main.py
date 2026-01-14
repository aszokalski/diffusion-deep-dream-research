import os
from pathlib import Path
from typing import Callable

import hydra
from loguru import logger
from omegaconf import OmegaConf
import submitit

from diffusion_deep_dream_research.config.config_schema import (
    ExperimentConfig,
    Stage,
    register_configs,
)
from diffusion_deep_dream_research.stages.s00_provision import run_provision
from diffusion_deep_dream_research.stages.s01_capture import run_capture
from diffusion_deep_dream_research.stages.s02_timestep_analysis import run_timestep_analysis
from diffusion_deep_dream_research.stages.s03_plots import run_plots
from diffusion_deep_dream_research.stages.s04_prior import run_prior
from diffusion_deep_dream_research.stages.s05_deep_dream import run_deep_dream
from diffusion_deep_dream_research.utils.logging import setup_distributed_logging

register_configs()

stages: dict[Stage, Callable[[ExperimentConfig], None]] = {
    Stage.provision: run_provision,
    Stage.capture: run_capture,
    Stage.timestep_analysis: run_timestep_analysis,
    Stage.plots: run_plots,
    Stage.prior: run_prior,
    Stage.deep_dream: run_deep_dream
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

    logger.info(f"Done! Results at: \n [absolute] {Path(os.getcwd())} \n [relative to outputs] {Path(os.getcwd()).relative_to(config.outputs_dir)}")




if __name__ == "__main__":
    main()