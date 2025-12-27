import torch
import torch.nn as nn
from unittest.mock import MagicMock, PropertyMock

from diffusion_deep_dream_research.model.hooks.base_hook import create_target_hook_context
from diffusion_deep_dream_research.model.hooks.steering_hook import SteeringHookFactory

from diffusion_deep_dream_research.model.modified_diffusion_pipeline_adapter import ModifiedDiffusionPipelineAdapter
from tests.mocks.mock_sae import MockSae

class TestLayerSteeringHook:
    def test_hook_captures_activation(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = SteeringHookFactory(
            sae=None,
            pipe_adapter=mock_adapter,
        )

        dummy_layer = nn.Conv2d(320, 320, 3)
        hook_context = create_target_hook_context(dummy_layer)

        fake_input = torch.randn(1, 320, 32, 32)

        with hook_context(hook_factory.create(
            channel=1,
            timesteps=[50],
            strength=123.0
        )) as hook:
            result = dummy_layer(fake_input).detach().numpy()

        assert result is not None
        # very loose testing to see if the correct channel was steered
        assert result[0,1,0,0] > 100
        assert result[0,2,0,0] < 100

class TestSaeSteeringHook:
    def test_hook_captures_activation(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_sae = MockSae(10, 100)

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = SteeringHookFactory(
            sae=mock_sae,
            pipe_adapter=mock_adapter,
        )

        dummy_layer = nn.Conv2d(10, 10, 3)
        hook_context = create_target_hook_context(dummy_layer)

        fake_input = torch.zeros(1, 10, 4, 4)

        with hook_context(hook_factory.create(
            channel=50,
            timesteps=[50],
            strength=123.0
        )) as hook:
            result = dummy_layer(fake_input).detach().numpy()


        assert result.shape == (1, 10, 2, 2) # convolution decreases spatial size

