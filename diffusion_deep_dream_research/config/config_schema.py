from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from diffusion_deep_dream_research.root import get_project_root


class AssetSourceType(str, Enum):
    huggingface = "huggingface"
    gdrive = "gdrive"

class AssetType(str, Enum):
    model = "model"
    dataset = "dataset"

@dataclass
class AssetConfig:
    name: str = MISSING
    source_type: AssetSourceType = MISSING
    url: str = MISSING
    path: Path = "${assets_dir}/${.name}" # pyright: ignore[reportAssignmentType]
    asset_type: AssetType = MISSING


class Stage(str, Enum):
    provision = "provision"
    capture = "capture"
    timestep_analysis = "timestep_analysis"
    plots = "plots"
    prior = "prior"

@dataclass
class StageConfig:
    name: str = MISSING

@dataclass
class ProvisionStageConfig(StageConfig):
    name: str = "provision"

@dataclass
class CaptureStageConfig(StageConfig):
    name: str = "capture"
    prompt_dataset: AssetConfig = MISSING
    num_images_per_prompt: int = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    log_every_n_steps: int = MISSING
    dev_n_prompts: Optional[int] = None

@dataclass
class TimestepAnalysisStageConfig(StageConfig):
    name: str = "timestep_analysis"
    capture_results_dir: Path = MISSING
    top_k: int = MISSING
    total_timesteps: int = MISSING
    peak_threshold: float = MISSING
    peak_separation: int = MISSING
    top_peak_count: int = MISSING

@dataclass
class PlotsStageConfig(StageConfig):
    name: str = "plots"
    timestep_analysis_results_dir: Path = MISSING
    frame_duration: float = MISSING

class PriorMethod(str, Enum):
    active_timesteps = "active_timesteps"
    all_timesteps = "all_timesteps"

@dataclass
class PriorStageConfig(StageConfig):
    name: str = "prior"
    timestep_analysis_results_dir: Path = MISSING
    start_channel: Optional[int] = None
    end_channel: Optional[int] = None
    method: PriorMethod = MISSING
    n_results: int = MISSING
    seeds: Optional[list[int]] = None
    steer_strength_scale: float = MISSING
    log_every_n_steps: int = MISSING

@dataclass
class FabricConfig:
    accelerator: str

@dataclass
class ExperimentConfig:
    project_name: str = "diffusion_deep_dream_research"
    experiment_name: str = "default_experiment"

    infrastructure_name: str = MISSING

    project_root: Path = get_project_root()

    data_root: str = MISSING

    assets_dir: Path = "${data_root}/assets"
    outputs_dir: Path = "${data_root}/outputs"

    models: dict[str, AssetConfig] = field(default_factory=dict)
    datasets: dict[str, AssetConfig] = field(default_factory=dict)
    stages: dict[Stage, StageConfig] = field(default_factory=dict)

    fabric: FabricConfig = MISSING

    model_to_analyse: AssetConfig = MISSING
    target_layer_name: str = MISSING
    sae: AssetConfig = MISSING
    use_sae: bool = MISSING

    stage: Stage = MISSING
    stage_config: StageConfig = "${stages.${stage}}"


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="base_config", node=ExperimentConfig)

    # Inspect schema with: python main.py infrastructure=schema --help
    cs.store(group="models", name="schema", node=AssetConfig)
    cs.store(group="datasets", name="schema", node=AssetConfig)
    cs.store(group="fabric", name="schema", node=FabricConfig)
    cs.store(group="model_to_analyse", name="schema", node=AssetConfig)
    cs.store(group="sae", name="schema", node=AssetConfig)
    cs.store(group="stages", name="provision_schema", node=ProvisionStageConfig)
    cs.store(group="stages", name="capture_schema", node=CaptureStageConfig)
    cs.store(group="stages", name="timestep_analysis_schema", node=TimestepAnalysisStageConfig)
    cs.store(group="stages", name="plots_schema", node=PlotsStageConfig)
    cs.store(group="stages", name="prior_schema", node=PriorStageConfig)
