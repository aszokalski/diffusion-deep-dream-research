"""
A script to preload all model checkpoints used in the experiments.
"""

import gdown
import os
from loguru import logger
from diffusion_deep_dream_research.config.general.general_config import general_config
from diffusion_deep_dream_research.config.general.models import GDriveModel, HFModel
from huggingface_hub import snapshot_download
from tqdm import tqdm

def main():
    os.makedirs(general_config.models.models_dir, exist_ok=True)

    for model in tqdm(general_config.models.models, desc="Downloading models"):
        model_path = str(os.path.join(general_config.models.models_dir, model.name))
        if os.path.exists(model_path):
            logger.info(f"{model.name} already exists. Skipping download.")
            continue

        logger.info(f"Downloading {model.name}...")
        if isinstance(model, GDriveModel):
            gdown.download_folder(url=model.url, output=model_path, quiet=False)
        elif isinstance(model, HFModel):
            snapshot_download(repo_id=model.repo_id, local_dir=model_path)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        logger.info(f"Downloaded {model.name} to {model_path}")


if __name__ == "__main__":
    main()