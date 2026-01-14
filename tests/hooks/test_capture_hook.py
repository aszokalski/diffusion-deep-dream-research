from unittest.mock import MagicMock, PropertyMock

import torch
import torch.nn as nn

from diffusion_deep_dream_research.core.hooks.base_hook import create_target_hook_context
from diffusion_deep_dream_research.core.hooks.capture_hook import CaptureHook, CaptureHookFactory
from diffusion_deep_dream_research.core.model.modified_diffusion_pipeline_adapter import (
    ModifiedDiffusionPipelineAdapter,
)
from tests.mocks.mock_sae import MockSae


class TestLayerCaptureHook:
    def test_hook_captures_activation(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = CaptureHookFactory(
            sae=None,
            pipe_adapter=mock_adapter,
        )

        dummy_layer = nn.Conv2d(320, 320, 3)
        hook_context = create_target_hook_context(dummy_layer)
        fake_input = torch.randn(1, 320, 32, 32)

        with hook_context(
            hook_factory.create(
                timesteps=[50],
                detach=True,
                early_exit=False,
            )
        ) as hook:
            dummy_layer(fake_input)

        activations = hook.get_last_activations()
        assert 50 in activations
        assert len(activations.keys()) == 1
        assert CaptureHook.ActivationType.RAW in activations[50]
        assert len(activations[50].keys()) == 1
        assert activations[50][CaptureHook.ActivationType.RAW].shape == (
            1,
            320,
        )
        assert activations[50][CaptureHook.ActivationType.RAW].requires_grad is False

    def test_hook_captures_activation_with_gradient(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = CaptureHookFactory(
            sae=None,
            pipe_adapter=mock_adapter,
        )

        dummy_layer = nn.Conv2d(320, 320, 3)
        hook_context = create_target_hook_context(dummy_layer)

        fake_input = torch.randn(1, 320, 32, 32)

        with hook_context(
            hook_factory.create(
                timesteps=[50],
                detach=False,
                early_exit=False,
            )
        ) as hook:
            dummy_layer(fake_input)

        activations = hook.get_last_activations()
        assert 50 in activations
        assert len(activations.keys()) == 1
        assert CaptureHook.ActivationType.RAW in activations[50]
        assert len(activations[50].keys()) == 1
        assert activations[50][CaptureHook.ActivationType.RAW].shape == (
            1,
            320,
        )
        assert activations[50][CaptureHook.ActivationType.RAW].requires_grad is True


class TestSaeCaptureHook:
    def test_hook_captures_activation_sae(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_sae = MockSae(320, 2000)

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = CaptureHookFactory(
            sae=mock_sae,
            pipe_adapter=mock_adapter,
        )

        dummy_layer = nn.Conv2d(320, 320, 3)
        hook_context = create_target_hook_context(dummy_layer)

        fake_input = torch.randn(1, 320, 32, 32)

        with hook_context(
            hook_factory.create(
                timesteps=[50],
                detach=True,
                early_exit=False,
            )
        ) as hook:
            dummy_layer(fake_input)

        activations = hook.get_last_activations()
        assert 50 in activations
        assert len(activations.keys()) == 1
        assert CaptureHook.ActivationType.RAW in activations[50]
        assert CaptureHook.ActivationType.ENCODED in activations[50]
        assert activations[50][CaptureHook.ActivationType.ENCODED].shape == (
            1,
            2000,
        )
        assert activations[50][CaptureHook.ActivationType.ENCODED].requires_grad is False

    def test_hook_captures_activation_sae_grad(self):
        mock_adapter = MagicMock(spec=ModifiedDiffusionPipelineAdapter)
        mock_pipe = MagicMock()
        mock_unet = MagicMock()
        mock_sae = MockSae(320, 2000)

        mock_unet.current_timestep = 50

        mock_pipe.unet = mock_unet
        type(mock_adapter).pipe = PropertyMock(return_value=mock_pipe)

        hook_factory = CaptureHookFactory(
            sae=mock_sae,
            pipe_adapter=mock_adapter,
        )

        dummy_layer = nn.Conv2d(320, 320, 3)
        hook_context = create_target_hook_context(dummy_layer)

        fake_input = torch.randn(1, 320, 32, 32)

        with hook_context(
            hook_factory.create(
                timesteps=[50],
                detach=False,
                early_exit=False,
            )
        ) as hook:
            dummy_layer(fake_input)

        activations = hook.get_last_activations()
        assert 50 in activations
        assert len(activations.keys()) == 1
        assert CaptureHook.ActivationType.RAW in activations[50]
        assert CaptureHook.ActivationType.ENCODED in activations[50]
        assert activations[50][CaptureHook.ActivationType.ENCODED].shape == (
            1,
            2000,
        )
        assert activations[50][CaptureHook.ActivationType.ENCODED].requires_grad is True
