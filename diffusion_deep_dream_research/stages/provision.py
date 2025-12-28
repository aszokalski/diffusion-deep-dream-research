import gdown
from huggingface_hub import snapshot_download
from tqdm import tqdm

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, ModelSourceType
from loguru import logger
import os

def run_provision(config: ExperimentConfig):
    logger.debug(f"Creating models directory at: {config.infrastructure.models_dir}")
    os.makedirs(config.infrastructure.models_dir, exist_ok=True)

    is_cluster = config.infrastructure.name == "athena"
    logger.debug(f"Running on cluster: {is_cluster}")
    for model in tqdm(config.models.values(), desc="Pulling models", mininterval=30 if is_cluster else 0.1):
        if os.path.exists(model.path):
            logger.info(f"{model.name} already exists. Skipping download.")
            continue

        logger.info(f"Downloading {model.name}...")

        if model.source_type == ModelSourceType.gdrive:
            gdown.download_folder(url=model.url, output=model.path, quiet=False)
        elif model.source_type == ModelSourceType.huggingface:
            snapshot_download(repo_id=model.url, local_dir=model.path)
        else:
            raise ValueError(f"Unknown model type: {model.source_type}")

        logger.info(f"Downloaded {model.name} to {model.path}")

    logger.info("Finished provisioning.")