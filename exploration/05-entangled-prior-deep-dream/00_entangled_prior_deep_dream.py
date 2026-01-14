from functools import lru_cache
import torch
import torch.nn as nn
import random
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Union, Sequence
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

# Diffusers imports
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention_processor import Attention
from torch.amp import GradScaler, autocast
import torchvision.transforms.functional as TF

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32
else:
    device = torch.device("cpu")
    dtype = torch.float32

print(f"Running on {device} with {dtype}")


# --- Helper Functions ---

def normalize_sequence(input_val: Union[int, Sequence[int], range]) -> List[int]:
    """Ensures input is always a list of integers."""
    if isinstance(input_val, int):
        return [input_val]
    if isinstance(input_val, range):
        return list(input_val)
    return list(input_val)


def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        latents = latents / pipe.vae.config.scaling_factor
        images = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        return Image.fromarray(images[0])


def show_comparison(img_steered, img_dreamed, title="Experiment Results", use_prior=True):
    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.imshow(img_steered)
    plt.title("Phase 1: Steered Prior" if use_prior else "Phase 1: Random Prior")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_dreamed)
    plt.title("Phase 2: Deep Dream Optimized")
    plt.axis('off')

    plt.show()


# --- Configuration ---

@dataclass
class ExperimentConfig:
    # Target Layer
    layer_name: str
    channels: Union[int, Sequence[int]]
    prompt: str = ""

    # Phase 1: Steering Params
    use_prior: bool = True
    steer_alpha: float = 10.0
    steer_timesteps: Union[int, Sequence[int]] = range(0, 1000)
    prior_noise_std: float = 0.0

    # Phase 2: Deep Dream Params
    dream_steps: int = 50
    dream_lr: float = 0.05
    dream_timesteps: Union[int, Sequence[int]] = 200
    see_through_step_noise: bool = True

    # Phase 2: Robustness Params
    jitter_max: int = 0  # Max pixels to shift (latents are 64x64, so 4 is significant)
    rotate_max: float = 0  # Degrees
    scale_max: float = 1.00 # Scale factor (1.0 = no scale)
    tv_penalty_weight: float = 0.01

    def __post_init__(self):
        assert self.prior_noise_std >= 0, "Prior noise std must be non-negative!"
        assert self.prior_noise_std < 1, "Prior noise > 1 is just noise. Just set use_prior=False"

# --- Pipeline Setup & Monkey Patching ---

model_path = "/models/style50"
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe = pipe.to(device)
pipe.safety_checker = None

original_forward = pipe.unet.forward


def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
    t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    pipe.unet.current_timestep = t_val
    return original_forward(sample, timestep, encoder_hidden_states, **kwargs)


pipe.unet.forward = intercepted_forward


# --- Core Logic Classes ---

class LatentAugmenter:
    """Handles differentiable augmentations on latents."""

    def __init__(self, jitter_max=0, rotate_max=0.0, scale_max=1.0):
        self.jitter_max = jitter_max
        self.rotate_max = rotate_max
        self.scale_max = scale_max

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        # 1. Jitter (Circular Roll) - Very effective for textures
        if self.jitter_max > 0:
            ox = random.randint(-self.jitter_max, self.jitter_max)
            oy = random.randint(-self.jitter_max, self.jitter_max)
            latents = torch.roll(latents, shifts=(ox, oy), dims=(2, 3))

        # 2. Affine (Rotation + Scale)
        # We skip this if params are zero to save compute/interpolation artifacts
        if self.rotate_max > 0 or self.scale_max != 1.0:
            angle = random.uniform(-self.rotate_max, self.rotate_max)
            scale = random.uniform(1.0, self.scale_max)

            # TF.affine expects [..., H, W]
            latents = TF.affine(
                latents,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR
            )

        return latents


@contextmanager
def steering_context(module: nn.Module, hook_fn):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield handle
    finally:
        handle.remove()


def create_steering_hook(alpha: float, target_timesteps: List[int], target_channels: List[int]):
    t_set = set(target_timesteps)
    c_indices = target_channels

    def hook(module, input, output):
        t = getattr(pipe.unet, 'current_timestep', None)
        if t is not None and t in t_set:
            if isinstance(module, Attention):
                output[:, :, c_indices] += alpha
            elif isinstance(module, Transformer2DModel):
                output[0][:, c_indices, :, :] += alpha
            else:
                output[:, c_indices, :, :] += alpha
        return output

    return hook


class UNetWrapper(nn.Module):
    def __init__(self, pipe: StableDiffusionPipeline, prompt: str):
        super().__init__()
        self.pipe = pipe
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.device = pipe.device
        self.dtype = pipe.unet.dtype
        self.prompt = prompt

    @lru_cache
    def embeddings(self, batch_size: int):
        with torch.no_grad():
            _, negative_embeds = self.pipe.encode_prompt(
                prompt=self.prompt,
                device=self.device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )
        return negative_embeds

    def forward(self, z: torch.Tensor, timestep: int):
        batch_size = z.shape[0]
        embeddings = self.embeddings(batch_size)

        return self.unet(
            sample=z,
            timestep=timestep,
            encoder_hidden_states=embeddings
        ).sample


class EarlyExit(Exception):
    pass


def compute_latent_tv_loss(latents: torch.Tensor) -> torch.Tensor:
    """
    Computes TV loss on the LATENT tensor (64x64).
    Prevents 'spiky' activations in semantic space.
    """
    # Calculate differences in height and width on the latent grid
    # latents shape: [1, 4, 64, 64]
    tv_h = torch.pow(latents[:, :, 1:, :] - latents[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(latents[:, :, :, 1:] - latents[:, :, :, :-1], 2).sum()
    return tv_h + tv_w


# --- Main Experiment Method ---

def run_experiment(config: ExperimentConfig):
    print(f"\n--- Starting Experiment ---")

    target_channels = normalize_sequence(config.channels)
    steer_timesteps = normalize_sequence(config.steer_timesteps)
    dream_timesteps = normalize_sequence(config.dream_timesteps)

    print(f"Target: {config.layer_name}")
    print(f"Optimizing {len(target_channels)} channels.")

    target_layer = dict(pipe.unet.named_modules())[config.layer_name]

    generator = torch.Generator(device).manual_seed(1024)

    if config.use_prior:
        # --- Phase 1: Generate Steered Prior ---
        print("Phase 1: Generating Steered Prior...")
        steering_fn = create_steering_hook(config.steer_alpha, steer_timesteps, target_channels)


        with steering_context(target_layer, steering_fn):
            output = pipe(
                config.prompt,
                num_inference_steps=50,
                output_type="latent",
                generator=generator
            )

        prior_latents = output.images
    else:
        print("Phase 1: Using Random Prior...")
        # use gaussian noise as prior
        prior_latents = torch.randn(
            (1, pipe.unet.config.in_channels, 64, 64),
            generator=generator,
            device=device,
            dtype=pipe.unet.dtype
        )


    # --- Phase 2: Deep Dream Optimization ---
    print(f"Phase 2: Optimizing with Deep Dream...")

    # Detach and enable gradients
    latents = prior_latents.clone().detach().to(device, dtype=dtype)

    if config.prior_noise_std > 0:
        #This way we preserve the prior variance
        signal_scale = (1 - config.prior_noise_std ** 2) ** 0.5
        noise_scale = config.prior_noise_std
        latents = (latents * signal_scale) + (torch.randn_like(latents) * noise_scale)

    image_prior = decode_latents(pipe, latents)

    latents.requires_grad_(True)


    optimizer = torch.optim.Adam([latents], lr=config.dream_lr)
    model_wrapper = UNetWrapper(pipe, config.prompt).eval()
    augmenter = LatentAugmenter(config.jitter_max, config.rotate_max, config.scale_max)

    activations = {}

    def capture_hook(module, input, output):
        activations["act"] = output
        raise EarlyExit

    #TODO: context manager
    hook_handle = target_layer.register_forward_hook(capture_hook)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    try:
        with tqdm(dream_timesteps, position=0, leave=False) as tbar:
            for timestep in tbar:
                tbar.set_description(f"Timestep: {timestep}")
                timestep_tensor = torch.tensor([timestep], device=device, dtype=dtype).int()

                with tqdm(range(config.dream_steps), position=1, leave=False) as pbar:
                    for _ in pbar:
                        optimizer.zero_grad()
                        activations.clear()

                        augmented_latents = augmenter(latents)

                        if config.see_through_step_noise:
                            noise = torch.randn_like(augmented_latents)
                            z = pipe.scheduler.add_noise(augmented_latents, noise, timestep_tensor)
                        else:
                            z = augmented_latents

                        with autocast(enabled=(device.type == "cuda"), device_type=device.type):
                            try:
                                # 2. Forward Pass on Augmented Latents
                                _ = model_wrapper(z, timestep)
                            except EarlyExit:
                                pass

                            act = activations["act"]

                            if isinstance(target_layer, Attention):
                                selected = act[:, :, target_channels]
                            elif isinstance(target_layer, Transformer2DModel):
                                selected = act[0][:, target_channels, :, :]
                            else:
                                selected = act[:, target_channels, :, :]

                            loss = -torch.mean(selected) + compute_latent_tv_loss(latents) * config.tv_penalty_weight

                        # 3. Backward Pass
                        if device.type == "cuda":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        pbar.set_description(f"Loss: {loss.item():.4f}")

    finally:
        hook_handle.remove()

    print("Decoding Final Result...")
    image_dream = decode_latents(pipe, latents)

    return image_prior, image_dream


# --- Execution ---

if __name__ == "__main__":
    pipe.scheduler.set_timesteps(50)
    all_timesteps = pipe.scheduler.timesteps.tolist()

    config = ExperimentConfig(
        layer_name="up_blocks.1.attentions.1",
        # layer_name="down_blocks.0.resnets.0.conv1",
        channels=range(1152, 1216),
        prompt="",

        use_prior=True,
        steer_alpha=50.0,
        steer_timesteps=all_timesteps,
        prior_noise_std=0.05,

        # Optimization Params
        dream_steps=200,
        dream_lr=0.01,
        dream_timesteps=500,
        see_through_step_noise=True,

        # # # New Robustness Params
        jitter_max=2,  # Shifts image by up to 4 pixels
        rotate_max=1.0,  # Rotates by up to 2 degrees
        scale_max=1.05,  # Scales by up to 1.05x
        tv_penalty_weight=0.000001
    )

    prior_img, dreamed_img = run_experiment(config)
    show_comparison(prior_img, dreamed_img, title=f"{config}", use_prior=config.use_prior)