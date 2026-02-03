# Visualizing Monosemantic Features in Diffusion Models
This is a code repository for my engineering thesis _Visualizing Monosemantic Features in Diffusion Models_.

## Requirements
- Conda
- Python 3.12
- Make

It's advised to run these experiments on a computing cluster as they require a lot of computation.
Some of the stages could run on a local machine but only if the results from the computationally heavy stages (ex. capture stage) are precomputed and available.

## Initial Setup
### Local
1. `make env`
2. `conda activate diffusion-deep-dream-research-env`
3. `make install`

### PLGrid
The PLGrid setup is complicated. I provide a few scripts to streamline the process. Note that this instruction is not universal for all SLURM clusters and is made specifically for PLGrid.
1. `chmod +x ./scripts/start-interactive-session.sh`
2. `chmod +x ./scripts/setup-plg.sh`
3. `./scripts/start-interactive-session.sh`
4. `source ./scripts/setup-plg.sh`
5. `make env`
6. `conda activate diffusion-deep-dream-research-env`
7. `make install`
8. `exit`
9. `source ./scripts/setup-plg.sh`
10. `conda activate diffusion-deep-dream-research-env`

Runnng these commands sets up a Conda environment in the `$SCRATCH` directory so it can be removed when the environment is unused for a few days. If that happens, just rerun the commands above.

#### Environment variables

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

The `.env` file configures user-specific settings used by the `athena` infrastructure config:
| Variable | Description |
|----------|-------------|
| `PLG_GROUP_NAME` | Your PLGrid group name |
| `PLG_USERNAME` | Your PLGrid username |
| `PLG_SLURM_PARTITION` | Your SLURM GPU partition |
| `PLG_SLURM_ACCOUNT` | Your SLURM account for GPU partition |
| `NOTIFICATION_EMAIL` | Email for SLURM job notifications |

## When you come back
### Local
1. `conda activate diffusion-deep-dream-research-env`

### PLGrid
1. `source ./scripts/setup-plg.sh`
2. `conda activate diffusion-deep-dream-research-env`

## Repository structure

```
├── conf/                                    # Hydra configuration
│   ├── config.yaml                          # Main config (composes all defaults)
│   ├── models/                              # Model asset definitions
│   ├── datasets/                            # Dataset asset definitions
│   ├── stages/                              # Per-stage hyperparameters
│   ├── infrastructure/                      # local vs athena (PLGrid) settings
│   └── transformation_robustness/           # Augmentation level presets
├── diffusion_deep_dream_research/           # Main source package
│   ├── config/config_schema.py              # Pydantic/dataclass config schema
│   ├── core/
│   │   ├── hooks/                           # CaptureHook, SteeringHook (PyTorch forward hooks)
│   │   ├── model/                           # HookedModelWrapper, ModifiedDiffusionPipelineAdapter
│   │   ├── data/                            # UniquePromptDataset, IndexDataset
│   │   └── regularisation/                  # Penalties, gradient transforms, latent augmentation
│   ├── stages/                              # Pipeline stage implementations (s00–s06)
│   └── utils/                               # Logging, result reading, torch helpers
├── scripts/                                 # PLGrid setup & inspector app
│   ├── inspector.py                         # Streamlit result inspector
│   ├── setup-plg.sh                         # PLGrid environment setup
│   └── start-interactive-session.sh         # PLGrid interactive session
├── submodules/SAeUron/                      # Sparse Autoencoder (git submodule)
├── notebooks/                               # Example notebooks
├── exploration/                             # Exploratory scripts
├── tests/                                   # Unit tests (pytest)
├── main.py                                  # Entry point
├── Makefile                                 # Build/install/test commands
└── pyproject.toml                           # Project metadata, dependencies, ruff config
```

## Experiment config

Configuration uses [Hydra](https://hydra.cc/) with type-safe schemas defined as Python dataclasses in `diffusion_deep_dream_research/config/config_schema.py`.

The main config file `conf/config.yaml` composes defaults from several groups:
- **models/** — asset definitions for Stable Diffusion, SAeUron, Style50 (name, source, download URL)
- **datasets/** — dataset assets (e.g. `unlearn_canvas`)
- **stages/** — hyperparameters for each pipeline stage
- **infrastructure/** — `local` (CPU, submitit_local) or `athena` (GPU A100, submitit_slurm)

Key top-level settings in `config.yaml`:
| Parameter | Description |
|-----------|-------------|
| `stage` | **Required.** Which pipeline stage to run (see below) |
| `experiment_name` | Name for the experiment run (affects output directory) |
| `model_to_analyse` | Which model to use (default: `style50`) |
| `target_layer_name` | UNet layer to hook into (default: `up_blocks.1.attentions.1`) |
| `use_sae` | Whether to use the Sparse Autoencoder |
| `infrastructure` | `local` or `athena` (default: `athena`)|

The `data_root` setting controls where all data lives. Two directories are derived from it:
- **`assets_dir`** = `<data_root>/assets` — downloaded models and datasets
- **`outputs_dir`** = `<data_root>/outputs` — experiment results

Each infrastructure config sets `data_root` differently:
- **`local`** — `data_root` is the project root directory
- **`athena`** — `data_root` is `$PLG_GROUPS_STORAGE/$PLG_GROUP_NAME/$PLG_USERNAME/<project_name>` (`$PLG_GROUPS_STORAGE` is preconfigured on PLGrid)

You can override it directly:
```bash
python main.py stage=capture data_root=/path/to/my/data
```

Override any parameter from the command line using Hydra syntax:
```bash
python main.py stage=capture experiment_name=my_experiment infrastructure=local
```

Later stages reference outputs from earlier stages via `*_results_dir` parameters in their config (e.g. `capture_results_dir`, `timestep_analysis_results_dir`, `prior_results_dir`). These are relative paths under the outputs directory.

### Infrastructure and GPU configuration

Each infrastructure config sets the Lightning Fabric accelerator and the Hydra launcher:
- **`local`** — `fabric.accelerator=cpu`, `submitit_local` launcher
- **`athena`** — `fabric.accelerator=cuda`, `submitit_slurm` launcher

Switch between them with the `infrastructure` override:
```bash
python main.py --multirun stage=capture infrastructure=local   # CPU, submitit_local
python main.py --multirun stage=capture infrastructure=athena  # A100 GPU, submitit_slurm
```

You can also override the accelerator directly:
```bash
python main.py stage=capture infrastructure=local fabric.accelerator=cuda
```

To run on a different cluster or machine, create a new config file in `conf/infrastructure/` (e.g. `my_cluster.yaml`) following the structure of `local.yaml` or `athena.yaml`, then use `infrastructure=my_cluster`.

The `athena` config (`conf/infrastructure/athena.yaml`) sets SLURM launcher parameters. To change GPU and node allocation, override the Hydra launcher settings:
```bash
python main.py --multirun stage=capture \
  infrastructure=athena \
  hydra.launcher.gpus_per_node=4 \
  hydra.launcher.tasks_per_node=4 \
  hydra.launcher.timeout_min=120
```

Key SLURM parameters in `athena.yaml` for package `hydra.launcher`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpus_per_node` | 0 | Number of GPUs per node |
| `tasks_per_node` | 1 | Number of tasks per node |
| `cpus_per_task` | 16 | CPU cores per task |
| `mem_gb` | 125 | Memory per node |
| `timeout_min` | 30 | Job time limit in minutes |

The athena config also sends email notifications on job BEGIN, END, and FAIL. Update the recipient with:
```bash
python main.py --multirun stage=capture \
  'hydra.launcher.additional_parameters.mail-user=your@email.com'
```


## Running experiments

All experiments are run through `main.py` with the `stage` parameter specifying which pipeline stage to execute:

```bash
python main.py --multirun stage=<stage_name>
```

The seven stages, meant to be run in order:

| Stage | Command | Description |
|-------|---------|-------------|
| **Provision** | `python main.py stage=provision` | Downloads models and datasets from HuggingFace/GDrive to `<output_dir>/assets/` |
| **Capture** | `python main.py stage=capture` | Runs inference on dataset prompts, captures neuron activations at each timestep. Supports distributed execution via Lightning Fabric. I recommend using at least 4 GPUs|
| **Timestep Analysis** | `python main.py stage=timestep_analysis` | Analyzes captured activations and computes active timesteps, activity peaks, and dataset examples. No GPU needed |
| **Plots** | `python main.py stage=plots` | Generates activity profile visualizations. No GPU needed.|
| **Prior** | `python main.py stage=prior` | Generates steered priors.  I recommend using at least 4 GPUs |
| **Deep Dream** | `python main.py stage=deep_dream` | Main optimization.  I recommend using at least 4 GPUs|
| **Representation** | `python main.py stage=representation` | Compiles final results into per-channel data shards and an index for inspection. No GPUs needed, |

### Multi-run (SLURM)

For cluster execution, use the `--multirun` flag along with `infrastructure=athena` (for PLGrid, default):

```bash
python main.py --multirun stage=capture
```

This command should be executed on a name node with the Conda environment setup as mentioned in the Initial Setup section.

Keep in mind that jobs executing with multi-run on a GPU partition do not have access to the internet. So the `provision` stage must be run either on the name node (just without the `--multirun` flag) or in an interactive session (`./scripts/start-interactive-session.sh`)

### Overriding stage parameters

Stage-specific parameters can be overridden directly:

```bash
python main.py stage=deep_dream stages.deep_dream.num_steps=200 stages.deep_dream.learning_rate=0.1
```

### Sweeps

Use Hydra's `--multirun` flag with comma-separated values or glob syntax to sweep over parameters:

```bash
python main.py --multirun stage=deep_dream \
  stages.deep_dream.learning_rate=0.01,0.05,0.1 \
  stages.deep_dream.total_variation_penalty_weight=0.1,0.5,1.0
```

Each combination launches a separate job. On SLURM (`infrastructure=athena`), jobs are submitted in parallel to the cluster. Locally, they run sequentially via `submitit_local`.

### Output directory structure

Results are written to `<data_root>/outputs/<experiment_name>/<stage>/<date>/<time>/`. Each run gets a unique timestamped directory. When using `--multirun`, an additional `multirun/` level is added with per-job subdirectories.

## Result inspector

A Streamlit app for browsing experiment results is available at `scripts/inspector.py`. It requires the representation stage to have been run first.

```bash
streamlit run scripts/inspector.py -- --base_dir <path_to_representation_output>
```

When running on PLGrid:

1. On PLG: `./scripts/start-interactive-session.sh`
2. Wait for the session to start. Note the node its running on ex. `t0048`
3. On your local machine: `ssh -L 8888:t0048:8888 plg<username>@athena.cyfronet.pl`
4. On PLG: `cd scripts`
5. On PLG: `streamlit run inspector.py --server.port 8888 -- --base_dir <path_to_representation_output>`
6. On your local machine open `localhost:8888` in your browser