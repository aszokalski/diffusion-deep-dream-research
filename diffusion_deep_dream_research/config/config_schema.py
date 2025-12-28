from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from diffusion_deep_dream_research.root import get_project_root


@dataclass
class InfrastructureConfig:
    name: str = MISSING

    project_root: Path = get_project_root()

    data_root: str = MISSING

    models_dir: Path = "${infrastructure.data_root}/models"
    outputs_dir: Path = "${infrastructure.data_root}/outputs"


class ModelSourceType(Enum):
    huggingface = "huggingface"
    gdrive = "gdrive"

@dataclass
class ModelConfig:
    name: str = MISSING
    source_type: ModelSourceType = MISSING
    url: str = MISSING
    path: Path = "${infrastructure.models_dir}/${.name}"

@dataclass
class Stage:
    name: str

@dataclass
class ProvisionStageConfig(Stage):
    name: str = "provision"

@dataclass
class ExperimentConfig:
    project_name: str = "diffusion_deep_dream_research"
    experiment_name: str = "default_experiment"

    infrastructure: InfrastructureConfig = MISSING
    models: dict[str, ModelConfig] = MISSING

    stage: Stage = MISSING

def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="base_config", node=ExperimentConfig)

    # Inspect schema with: python main.py infrastructure=schema --help
    cs.store(group="infrastructure", name="schema", node=InfrastructureConfig)
    cs.store(group="models", name="schema", node=ModelConfig)
    cs.store(group="stage", name="schema", node=ProvisionStageConfig)