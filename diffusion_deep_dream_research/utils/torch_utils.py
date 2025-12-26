from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModel

import torch.nn as nn
import torch

def reshape_to_batch_spatial_channels(module: nn.Module, activations: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the activations to (batch_size, spatial_dim, channels)
    """
    if isinstance(module, Attention):
        # (batch_size, w*h, channels)
        reshaped_activations = activations
    elif isinstance(module, Transformer2DModel):
        # (batch_size, channels, h, w)
        b, c, h, w = activations.shape
        reshaped_activations = activations.permute(0, 2, 3, 1).reshape(b, h * w, c)
    elif isinstance(module, nn.Conv2d):
        # (batch_size, channels, h, w)
        b, c, h, w = activations.shape
        reshaped_activations = activations.permute(0, 2, 3, 1).reshape(b, h * w, c)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}. Size of activations: {activations.shape}")

    return reshaped_activations