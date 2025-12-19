from pydantic_settings import BaseSettings, SettingsConfigDict

from diffusion_deep_dream_research.config.general.models import ModelsConfig


class GeneralConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    models: ModelsConfig = ModelsConfig()



general_config = GeneralConfig()