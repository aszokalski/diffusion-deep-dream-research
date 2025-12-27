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
        return activations
    elif isinstance(module, (Transformer2DModel, nn.Conv2d)):
        # (batch_size, channels, h, w)
        b, c, h, w = activations.shape
        return activations.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}. Size of activations: {activations.shape}")


def restore_from_batch_spatial_channels(module: nn.Module,
                                        activations: torch.Tensor,
                                        original_shape: torch.Size) -> torch.Tensor:
    """
    Restores activations from (batch_size, spatial_dim, channels) back to original shape.
    Inverse of reshape_to_batch_spatial_channels.
    """
    b, s, c = activations.shape

    if isinstance(module, Attention):
        # (batch_size, w*h, channels)
        return activations.view(original_shape)

    elif isinstance(module, (Transformer2DModel, nn.Conv2d)):
        # (batch_size, channels, h, w)
        h, w = original_shape[2], original_shape[3]
        restored = activations.view(b, h, w, c)
        return restored.permute(0, 3, 1, 2).contiguous()

    else:
        raise ValueError(f"Unsupported module type: {type(module)}")