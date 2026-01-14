import torch
from matplotlib import pyplot as plt
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention_processor import Attention
from dataclasses import dataclass

from contextlib import contextmanager
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    dtype = torch.float16
else:
    dtype = torch.float32


model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained("/models/style50", torch_dtype=dtype)
pipe = pipe.to(device)
pipe.safety_checker = None

original_forward = pipe.unet.forward

def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
    pipe.unet.current_timestep = timestep.item() if isinstance(timestep, torch.Tensor) else timestep

    return original_forward(sample, timestep, encoder_hidden_states, **kwargs)

pipe.unet.forward = intercepted_forward

@contextmanager
def hook_context(module, hook_fn):
    handle = module.register_forward_hook(hook_fn)

    try:
        yield handle
    finally:
        handle.remove()

def create_steering_hook(alpha: float, t_range: tuple[int, int], channel_range: tuple[int, int]):
    def steer_activations_hook(module, input, output):
        # print(f"Steering activations of shape {output.shape}")
        print(module.__class__)
        print(input[0].shape)
        print(output[0].shape)
        t = pipe.unet.current_timestep

        if t_range[0] <= t <= t_range[1]:
            print(f"Steering at timestep {t}")
            if isinstance(module, Attention):
                print("Att")
                # (batch, latent_w*latent_h, channels [embedding dim])
                output[:, :, channel_range[0]:channel_range[1]] += alpha
                # output += alpha
            elif isinstance(module, Transformer2DModel):
                # (batch, channels [embedding dim], latent_w, latent_h)
                output[0][:, channel_range[0]:channel_range[1], :, :] += alpha
            else:
                # (batch, channels, latent_w, latent_h)
                output[:, channel_range[0]:channel_range[1], :, :] += alpha

        return output
    return steer_activations_hook

generator = torch.Generator(device).manual_seed(1024) #12345
latents = torch.randn(
    (1, pipe.unet.config.in_channels, 64, 64),
    generator=generator,
    device=device,
    dtype=pipe.unet.dtype
)

@dataclass(frozen=True)
class E01Config:
    layer_name: str
    channel_range: tuple[int, int]
    alpha: float = 10.0
    prompt: str = ""
    t_range: tuple[int, int] = (0, 1000)


from functools import lru_cache

@lru_cache(maxsize=None)
def cached_pipe(prompt: str):
    return pipe(
        prompt=prompt,
        latents=latents,
        num_inference_steps=50,
        guidance_scale=1.0
    ).images[0]

def experiment_e01(config: E01Config):
    image_no_steering = cached_pipe(config.prompt)

    target_layer = dict(pipe.unet.named_modules())[config.layer_name]

    with hook_context(target_layer, create_steering_hook(config.alpha, config.t_range, config.channel_range)):
        image_steering = pipe(
            config.prompt,
            latents=latents,
            num_inference_steps=50,
            guidance_scale=1.0
        ).images[0]

    diff = np.abs(np.array(image_steering).astype(np.int16) - np.array(image_no_steering).astype(np.int16)).astype(np.uint8)
    diff_pil = Image.fromarray(diff)

    return image_no_steering, image_steering, diff_pil

configs = [
    # Slice 1: Channels 0-64
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(0, 32), alpha=15.0),

    # Slice 2: Channels 64-128
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(32, 64), alpha=15.0),

    # Slice 3: Channels 128-192
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(64, 96), alpha=15.0),

    # Slice 4: Channels 192-256
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(128, 160), alpha=15.0),

    # Slice 5: Channels 256-320
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(160, 192), alpha=15.0),

    # Slice 6: Channels 320-384
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(192, 224), alpha=15.0),

    # Slice 7: Channels 384-448
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(224, 256), alpha=15.0),

    # Slice 8: Channels 448-512
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(448, 512), alpha=15.0),

    # Slice 9: Channels 512-576
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(512, 576), alpha=15.0),

    # Slice 10: Channels 576-640
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(576, 640), alpha=15.0),

    # Slice 11: Channels 640-704
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(640, 704), alpha=15.0),

    # Slice 12: Channels 704-768
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(704, 768), alpha=15.0),

    # Slice 13: Channels 768-832
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(768, 832), alpha=15.0),

    # Slice 14: Channels 832-896
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(832, 896), alpha=15.0),

    # Slice 15: Channels 896-960
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(896, 960), alpha=15.0),

    # Slice 16: Channels 960-1024
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(960, 1024), alpha=15.0),

    # Slice 17: Channels 1024-1088
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(1024, 1088), alpha=15.0),

    # Slice 18: Channels 1088-1152
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(1088, 1152), alpha=15.0),

    # Slice 19: Channels 1152-1216
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(1152, 1216), alpha=15.0),

    # Slice 20: Channels 1216-1280 (The final slice)
    E01Config(layer_name="up_blocks.1.attentions.1", channel_range=(1216, 1280), alpha=15.0),
]

for config in configs:
    image_no_steering, image_steering, diff_pil = experiment_e01(config)

    plt.figure(figsize=(10, 5))
    plt.suptitle(config.__str__())
    plt.subplot(1, 3, 1)
    plt.imshow(image_no_steering)
    plt.title("No steering")
    plt.subplot(1, 3, 2)
    plt.imshow(image_steering)
    plt.title("Steering")
    plt.subplot(1, 3, 3)
    plt.imshow(diff_pil)
    plt.title("Difference")
    plt.show()
