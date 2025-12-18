import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn as nn
from diffusers import PNDMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from matplotlib import pyplot as plt
from torch.amp import GradScaler, autocast
from tqdm import tqdm

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

def to_chw(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

def to_hwc(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    max_val = tensor.max()
    min_val = tensor.min()

    if max_val == min_val:
        return torch.zeros_like(tensor)

    norm_0_1 = (tensor - min_val) / (max_val - min_val)

    return norm_0_1 * 2.0 - 1.0

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    max_val = tensor.max()
    min_val = tensor.min()

    if max_val == min_val:
        return torch.zeros_like(tensor)
    tensor = (tensor - min_val) / (max_val - min_val)

    return tensor * 255.0

def show_images(images: torch.Tensor, size=(15, 15), title=None, save_path=None) -> list[torch.Tensor]:
    """
    Shows / saves an image
    returns a list of processed images
    """
    images = denormalize(images)

    plt.figure(figsize=size)

    if title is not None:
        plt.suptitle(title, fontsize=16, y=0.6)

    res_images = []
    for i in range(images.shape[0]):
        img = to_hwc(images)[i].cpu().int().numpy()
        res_images.append(img)
        plt.subplot(1, images.shape[0], i + 1)
        plt.imshow(img)
        plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return res_images

class UNetWrapper(nn.Module):
    def __init__(self, pipe: StableDiffusionPipeline, timestep: int, prompt: str = "", use_noise: bool = True):
        super().__init__()
        self.pipe = pipe
        self.unet: UNet2DConditionModel = pipe.unet
        self.scheduler: PNDMScheduler = pipe.scheduler
        self.device = pipe.device
        self.dtype = pipe.unet.dtype
        self.prompt = prompt
        self.timestep = timestep
        self.use_noise = use_noise

    @lru_cache
    def embeddings(self, batch_size: int):
        with torch.no_grad():
            _, negative_embeds = self.pipe.encode_prompt(
                prompt="",
                device=self.device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )
        return negative_embeds

    @lru_cache
    def timestep_tensor(self, timestep):
        return torch.tensor([timestep], device=self.device, dtype=self.dtype).int()


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet with the given timestep and prompt embeddings.
        :param z: Latent input tensor of shape (batch_size, channels, height, width)
        :return: Latent output tensor of shape (batch_size, channels, height, width)
        """
        z = z.to(self.device, dtype=self.dtype)

        batch_size = z.shape[0]
        embeddings = self.embeddings(batch_size)
        c = z.shape[1]
        h = z.shape[2]
        w = z.shape[3]
        if self.use_noise:
            noise = torch.randn((batch_size, c, h, w), device=pipe.device, dtype=pipe.dtype)
            z_noise = self.scheduler.add_noise(z, noise, self.timestep_tensor(self.timestep))

        return self.unet(
            sample=z_noise if self.use_noise else z,
            timestep=self.timestep,
            encoder_hidden_states=embeddings
        ).sample

def encode_images(images: torch.Tensor) -> torch.Tensor:
    """
    Encode images into latent space using the VAE encoder.
    :param images: Input images tensor of shape (batch_size, channels, height, width)
    :return: Latent representation tensor of shape (batch_size, latent_channels, latent_height, latent_width)
    """
    with torch.no_grad():
        latent_dist = pipe.vae.encode(images).latent_dist
        latents = latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    return latents

def decode_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Decode latent representations back into images using the VAE decoder.
    :param latents: Latent representation tensor of shape (batch_size, latent_channels, latent_height, latent_width)
    :return: Decoded images tensor of shape (batch_size, channels, height, width)
    """
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        # Cast latents to the VAE's dtype (float16) before decoding
        images = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    return images

model_test = UNetWrapper(pipe, timestep=1).to(device).eval()

@dataclass(frozen=True)
class E0Config:
    layer_name: str
    channel: int | Sequence[int]
    latent_size: int
    batch_size: int
    timestep: int | Sequence[int]
    no_noise: bool
    set_steps: int = None
    set_lr: float = None
    prior: torch.Tensor = None

    @property
    def layer_depth(self) -> float:
        """
        Returns the depth of the layer as a percentage of the total number of layers [0, 1].
        :return:
        """
        block = self.layer_name.split(".")[0]
        if block == "down_blocks":
            return 0.2
        elif block == "mid_blocks":
            return 0.6
        else:
            return 0.8


    @property
    def steps(self) -> int:
        if self.set_steps is not None:
            return self.set_steps
        return math.floor(50 + 250 * self.layer_depth)

    @property
    def lr(self) -> float:
        if self.set_lr is not None:
            return self.set_lr
        return 0.1 ** (self.layer_depth + 1)



def get_latents(batch_size: int, latent_size: int, device) -> torch.Tensor:
    #cache for consistent results in tests
    return torch.randn(batch_size, 4, latent_size, latent_size, device=device, dtype=torch.float32) * 0.01

def experiment_e0(config: E0Config, model: UNetWrapper) -> torch.Tensor:
    model.use_noise = True
    if config.prior:
        latents = encode_images(config.prior)
    else:
        latents = get_latents(config.batch_size, config.latent_size, model.device)
    latents.requires_grad_(True)

    optimizer = torch.optim.Adam([latents], lr=config.lr)

    is_cuda_half = (model.device.type == "cuda" and dtype == torch.float16)
    scaler = GradScaler(enabled=is_cuda_half)

    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    target_layer = dict(model.unet.named_modules())[config.layer_name]
    hook_handle = target_layer.register_forward_hook(get_activation("target"))
    print(f"Optimizing Channel {config.channel} on {config.layer_name} at timestep {config.timestep} for {config.steps} steps with lr {config.lr}...")

    if isinstance(config.channel, int):
        channels = [config.channel]
    else:
        channels = config.channel

    if isinstance(config.timestep, int):
        timesteps = [config.timestep]
    else:
        timesteps = config.timestep

    try:
        with tqdm(timesteps, position=0, leave=False) as tbar:
            for timestep in tbar:
                tbar.set_description(f"Timestep: {timestep}")
                model.timestep = timestep
                with tqdm(range(config.steps), position=1, leave=False) as pbar:
                    for i in pbar:
                        activations.clear()

                        optimizer.zero_grad(set_to_none=True)

                        with autocast(enabled=is_cuda_half, device_type=device.type):
                            _ = model(latents)

                            if "target" not in activations:
                                raise RuntimeError(f"Hook failed to trigger on step {i}. The layer '{config.layer_name}' might be skipped in the forward pass.")

                            act = activations["target"]


                            if isinstance(target_layer, Attention):
                                # (batch, latent_w*latent_h, channels [embedding dim])
                                loss = -torch.mean(act[:, :, channels])
                            elif isinstance(target_layer, Transformer2DModel):
                                # (batch, channels [embedding dim], latent_w, latent_h)
                                loss = -torch.mean(act[0][:, channels, :, :])
                            else:
                                # (batch, channels, latent_w, latent_h)
                                loss = -torch.mean(act[:, channels, :, :])

                        # Use scaler only if it was instantiated (i.e., on CUDA)
                        if is_cuda_half:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        pbar.set_description(f"Loss: {loss.item():.4f}")
    finally:
        hook_handle.remove()
        print("Hook removed.")

    print("Decoding final result...")
    final_images = decode_latents(latents.detach())
    return final_images




configs = [
    # E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224,256), latent_size=16, batch_size=5,
    #          timestep=[500, 600, 700, 800, 999], no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(768, 832), latent_size=16, batch_size=5,
             timestep=500, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224, 256), latent_size=16, batch_size=5,
             timestep=500, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224, 256), latent_size=16, batch_size=5,
             timestep=600, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224, 256), latent_size=16, batch_size=5,
             timestep=700, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224, 256), latent_size=16, batch_size=5,
             timestep=800, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224, 256), latent_size=16, batch_size=5,
             timestep=900, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1", channel=range(224, 256), latent_size=16, batch_size=5,
             timestep=999, no_noise=False),
    # E0Config(layer_name="up_blocks.1.attentions.1.transformer_blocks.0.attn1", channel=(1152, 1216), latent_size=16, batch_size=5, timestep=500, no_noise=False),
    E0Config(layer_name="up_blocks.1.attentions.1.transformer_blocks.0.attn1", channel=(1152, 1216), latent_size=16, batch_size=5,
             timestep=200, no_noise=False),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=[0, 100, 400, 800, 999], no_noise=False),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=[999, 800, 400, 100, 0], no_noise=False),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=0, no_noise=True),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=100, no_noise=False),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=100, no_noise=True),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=200, no_noise=False),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=200, no_noise=True),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=800, no_noise=False),
    E0Config(layer_name="down_blocks.0.resnets.0.conv2", channel=5, latent_size=16, batch_size=5, timestep=800, no_noise=True)
]

for i, config in enumerate(configs):
    print(f"EXPERIMENT {i+1}/{len(configs)}")
    print(config)

    try:
        final_image = experiment_e0(config, model_test)
    except Exception as e:
        print(f"Error in experiment_e0: {e}")
        continue

    title = f"Layer: {config.layer_name}, Channel: {config.channel}, Timestep: {config.timestep}, Noise: {not config.no_noise}"
    key=f"{config.layer_name}-ch{config.channel}-t{config.timestep}-it{config.steps}-lr{config.lr}" + ("-no-noise" if config.no_noise else "")
    result = show_images(final_image, size=(20,20), title=title)
