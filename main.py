from typing import Callable

import hydra
from omegaconf import OmegaConf

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, register_configs
from diffusion_deep_dream_research.stages.provision import run_provision
from loguru import logger

register_configs()

stages: dict[str, Callable[[ExperimentConfig], None]] = {
    "provision": run_provision
}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    config: ExperimentConfig = OmegaConf.to_object(cfg)
    logger.info("Running with config:")
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"Executing stage: {config.stage.name}...")
    stages[config.stage.name](
        config
    )
    logger.info("Done!")




if __name__ == "__main__":
    main()