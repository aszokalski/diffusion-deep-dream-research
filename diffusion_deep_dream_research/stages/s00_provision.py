import gdown
from huggingface_hub import snapshot_download
from tqdm import tqdm

from diffusion_deep_dream_research.config.config_schema import ExperimentConfig, AssetSourceType, AssetType
from loguru import logger
import os

def run_provision(config: ExperimentConfig):
    logger.debug(f"Creating models directory at: {config.assets_dir}")
    os.makedirs(config.assets_dir, exist_ok=True)

    is_cluster = config.infrastructure_name == "athena"
    logger.debug(f"Running on cluster: {is_cluster}")
    for asset in tqdm((config.models | config.datasets).values(), desc="Pulling models and datasets", mininterval=30 if is_cluster else 0.1):
        if os.path.exists(asset.path):
            logger.info(f"{asset.name} already exists. Skipping download.")
            continue

        logger.info(f"Downloading {asset.name}...")

        if asset.source_type == AssetSourceType.gdrive:
            if 'folders' in asset.url:
                gdown.download_folder(url=asset.url, output=str(asset.path), quiet=False)
            else:
                gdown.download(url=asset.url, output=str(asset.path), quiet=False, fuzzy=True)
        elif asset.source_type == AssetSourceType.huggingface:
            snapshot_download(repo_id=asset.url, local_dir=asset.path, repo_type=asset.asset_type.value)
        else:
            raise ValueError(f"Unknown model type: {asset.source_type}")

        logger.info(f"Downloaded {asset.name} to {asset.path}")

    logger.info("Finished provisioning.")