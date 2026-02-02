from dataclasses import fields, replace

from diffusion_deep_dream_research.config.config_schema import DeepDreamStageConfig


def resolve_sae_config(config: DeepDreamStageConfig) -> DeepDreamStageConfig:
    """
    Returns a new config instance for sae. Standard parameters
    are replaced by their _sae counterparts if the sae value is not None.
    """

    updates = {}

    for field in fields(config):
        if field.name.endswith("_sae"):
            sae_value = getattr(config, field.name)

            if sae_value is not None:
                # Remove '_sae' to find the target base name
                base_name = field.name[:-4]
                if hasattr(config, base_name):
                    updates[base_name] = sae_value

    return replace(config, **updates)