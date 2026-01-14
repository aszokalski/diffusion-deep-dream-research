# diffusion-deep-dream-research

## Initial Setup
### Local
1. `make env`
2. `conda activate diffusion-deep-dream-research-env`
3. `make install`

### PLGrid
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

## When you come back
### Local
1. `conda activate diffusion-deep-dream-research-env`

### PLGrid
1. `source ./scripts/setup-plg.sh`
2. `conda activate diffusion-deep-dream-research-env`