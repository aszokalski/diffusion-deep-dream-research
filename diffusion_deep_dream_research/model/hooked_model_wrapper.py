import lightning as L
from diffusers import DiffusionPipeline


class HookedModelWrapper(L.LightningModule):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__()