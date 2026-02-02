# Visualizing Monosemantic Features in Diffusion Models
This is a code repository for my engineering thesis _Visualizing Monosemantic Features in Diffusion Models_.

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

## When you come back
### Local
1. `conda activate diffusion-deep-dream-research-env`

### PLGrid
1. `source ./scripts/setup-plg.sh`
2. `conda activate diffusion-deep-dream-research-env`

## Repository structure

## Experiment config

## Running experiments


## Result inspector