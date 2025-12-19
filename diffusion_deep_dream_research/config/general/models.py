from pydantic import BaseModel, Field

from diffusion_deep_dream_research.root import get_project_root


class Model(BaseModel):
    name: str

class GDriveModel(Model):
    url: str

class HFModel(Model):
    repo_id: str

class ModelsConfig(BaseModel):
    models_dir: str = Field(default="models")

    @property
    def models_dir_abs(self) -> str:
        return str(get_project_root().joinpath(self.models_dir))

    models: list[Model] = [
        GDriveModel(name="style50", url="https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx?usp=share_link"),
        HFModel(name="stable-diffusion-v1-4", repo_id="CompVis/stable-diffusion-v1-4"),
        HFModel(name="SAeUron", repo_id="bcywinski/SAeUron")
        ]

