from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

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
    path: Path = "${assets_dir}/${.name}"  # type: ignore[reportAssignmentType]
    asset_type: AssetType = MISSING


class Stage(str, Enum):
    provision = "provision"
    capture = "capture"
    timestep_analysis = "timestep_analysis"
    plots = "plots"
    prior = "prior"
    deep_dream = "deep_dream"


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


class Timesteps(str, Enum):
    active_timesteps = "active_timesteps"
    all_timesteps = "all_timesteps"
    activity_peaks = "activity_peaks"


@dataclass
class PriorStageConfig(StageConfig):
    name: str = "prior"
    timestep_analysis_results_dir: Path = MISSING

    start_channel: Optional[int] = None
    end_channel: Optional[int] = None
    timesteps: Timesteps = MISSING
    n_results: int = MISSING
    seeds: Optional[list[int]] = None
    steer_strength_scale: float = MISSING
    steer_strength_scale_sae: Optional[float] = None
    log_every_n_steps: int = MISSING


@dataclass
class DeepDreamStageConfig(StageConfig):
    # SAE parameters are meant to be used
    # if experiments prove sae needs different values.
    # First experiments will use only the normal parameters to comapare.
    # Only the parameters which we intend to test have sae variants.

    name: str = "deep_dream"
    timestep_analysis_results_dir: Path = MISSING

    # We will use one set of results for SAE and
    # non sae so they need to have used correct
    # hyperparams for each.
    prior_results_dir: Path = MISSING

    # These are just for small scale sweeps:
    start_channel: Optional[int] = None
    end_channel: Optional[int] = None
    channels: Optional[list[int]] = None
    channels_sae: Optional[list[int]] = None

    timesteps: list[Union[int, str]] = MISSING
    use_just_one_timestep: Optional[bool] = False

    # This is not really a regularization,
    # but it is similar to transformation robustness.
    see_through_schedule_noise: bool = MISSING

    # REGULARISATION PARAMETERS
    use_prior: bool = MISSING

    total_variation_penalty_weight: float = MISSING
    total_variation_penalty_weight_sae: Optional[float] = None

    range_penalty_weight: float = MISSING
    range_penalty_weight_sae: Optional[float] = None
    range_penalty_threshold: float = 3.0

    moment_penalty_weight: float = MISSING
    moment_penalty_weight_sae: Optional[float] = None

    gradient_smoothing_sigma_start: float = MISSING
    gradient_smoothing_sigma_end: float = MISSING

    gradient_smoothing_kernel_size: int = 9

    use_decorrelated_space: bool = MISSING
    use_decorrelated_space_sae: Optional[float] = None

    jitter_max: int = MISSING
    jitter_max_sae: Optional[int] = None

    rotate_max: float = MISSING
    rotate_max_sae: Optional[float] = None

    scale_max: float = MISSING
    scale_max_sae: Optional[float] = None

    # OPTIMIZATION PARAMETERS
    num_steps: int = MISSING
    num_steps_sae: Optional[int] = None

    learning_rate: float = MISSING
    learning_rate_sae: Optional[float] = None

    # Only used if not using prior.
    # Otherwise, using the number of results from prior.
    seeds: Optional[list[int]] = None
    n_results: Optional[int] = None

    log_every_n_steps: int = MISSING
    intermediate_opt_results_every_n_steps: int = MISSING


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

    assets_dir: str = "${data_root}/assets"
    outputs_dir: str = "${data_root}/outputs"

    models: dict[str, AssetConfig] = field(default_factory=dict)
    datasets: dict[str, AssetConfig] = field(default_factory=dict)
    stages: dict[Stage, StageConfig] = field(default_factory=dict)

    fabric: FabricConfig = MISSING

    model_to_analyse: AssetConfig = MISSING
    target_layer_name: str = MISSING
    sae: AssetConfig = MISSING
    use_sae: bool = MISSING

    stage: Stage = MISSING
    stage_config: StageConfig = "${stages.${stage}}"  # type: ignore[reportAssignmentType]


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
    cs.store(group="stages", name="deep_dream_schema", node=DeepDreamStageConfig)
