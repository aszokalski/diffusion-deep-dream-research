import torch
import torch.nn as nn
from unittest.mock import MagicMock, PropertyMock

from diffusion_deep_dream_research.model.hooks.base_hook import hook_context
from diffusion_deep_dream_research.model.hooks.capture_hook import CaptureHookFactory, CaptureHook

from diffusion_deep_dream_research.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter


class TestCaptureHook:
    def test_hook_captures_activation_mocked(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = CaptureHookFactory(
            sae=None,
            detach=True,
            early_exit=False,
            timesteps=[50],
            pipe_adapter=mock_adapter,
        )


        dummy_layer = nn.Conv2d(320, 320, 3)

        fake_input = torch.randn(1, 320, 32, 32)

        with hook_context(dummy_layer, hook_factory.create()) as hook:
            dummy_layer(fake_input)

        activations = hook.get_last_activations()
        assert 50 in activations
        assert activations[50].shape == (320,)