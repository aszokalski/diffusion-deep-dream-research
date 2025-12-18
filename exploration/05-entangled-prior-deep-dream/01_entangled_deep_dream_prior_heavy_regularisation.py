from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Union, Sequence, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import math
import numpy as np

# Diffusers imports
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention_processor import Attention
from torch.amp import GradScaler, autocast
import torchvision.transforms.functional as TF

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

# Helpers
def normalize_sequence(input_val: Union[int, Sequence[int], range]) -> List[int]:
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
    plt.title("Phase 1: Prior")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_dreamed)
    plt.title("Phase 2: Decorrelated Deep Dream")
    plt.axis('off')
    plt.show()

# Regularisations

def compute_tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Total Variation regularization.
    Vertical and Horizontal gradients ^2 sum
    https://en.wikipedia.org/wiki/Total_variation

    Probably should calculate mean not sum
    """
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return h_tv + w_tv


def gaussian_blur_gradient(grad: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Gradient blurring
    """
    if sigma <= 0: return grad
    if kernel_size % 2 == 0: kernel_size += 1
    return TF.gaussian_blur(grad, kernel_size=kernel_size, sigma=sigma)


def get_preconditioner(h, w, device):
    """
    Preconditioning matrix for FFT-based decorrelation.
    """
    # frequency grid
    Y, X = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")

    # Centering the frequencies
    X = X - w // 2
    Y = Y - h // 2

    # Calculate distance from DC component (0,0)
    # The 'epsilon' prevents division by zero at the center
    freq_sq = X ** 2 + Y ** 2
    freqs = torch.sqrt(freq_sq.float())

    # The scaling factor is 1 / freq.
    # We add 1.0 to avoid exploding the DC component (global color).
    scale = 1.0 / (freqs + 1.0)

    # Shift back to corner-centered for FFT compatibility
    scale = torch.fft.ifftshift(scale)

    return scale


def precondition_gradient(grad: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
    """
    Applies Decorrelated Space Preconditioning via FFT.
    High frequencies scaled down
    """

    # 1. FFT to move to frequency domain
    grad_fft = torch.fft.fft2(grad.float())

    # 2. Scale frequencies (Whitening)
    # Broadcast scale_matrix [H, W] to [B, C, H, W]
    grad_fft_scaled = grad_fft * scale_matrix.unsqueeze(0).unsqueeze(0)

    # 3. Inverse FFT to move back to spatial domain
    grad_decorrelated = torch.fft.ifft2(grad_fft_scaled).real

    # 4. Normalize magnitude to match original gradient roughly
    # This keeps the learning rate interpretable
    norm_factor = grad.std() / (grad_decorrelated.std() + 1e-8)

    return grad_decorrelated.to(grad.dtype) * norm_factor

def gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Blurs a tensor (Latent or Image).
    kernel_size must be odd.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)


def compute_range_loss(latents: torch.Tensor, threshold: float = 2.5) -> torch.Tensor:
    """
    Penalizes values that exceed the statistical 'safe zone' of the VAE.
    Instead of hard clamping, we apply a quadratic penalty to outliers.
    Safe zone for Unit Variance is typically [-3, 3].
    """
    # Calculate magnitude
    magnitude = torch.abs(latents)

    # Identify how far 'out of bounds' the values are
    # ReLU ensures we only punish values > threshold
    out_of_bounds = F.relu(magnitude - threshold)

    # Quadratic penalty ensures strong pull-back for extreme outliers
    return torch.mean(out_of_bounds ** 2)


def compute_moment_loss(latents: torch.Tensor) -> torch.Tensor:
    """
    Enforces the 'Standard Normal' assumption (Mean=0, Std=1).
    Prevents the entire image from drifting into a specific color (mean shift)
    or becoming too washed out/contrasty (variance shift).
    """
    # 1. Mean Penalty (Keep it centered at 0)
    mean_loss = torch.norm(latents.mean())

    # 2. Variance Penalty (Keep std dev close to 1.0)
    # The VAE expects Unit Variance.
    std_loss = torch.norm(latents.std() - 1.0)

    return mean_loss + std_loss

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
    prior_blur_sigma: float = 0.0
    prior_blur_kernel_size: int = 9

    # Phase 2: Deep Dream Optimization Params
    num_steps: int = 500
    learning_rate: float = 0.05
    see_through_schedule_noise: bool = True

    # Annealing Schedule (High t -> Low t)
    start_timestep: int = 600
    end_timestep: int = 100

    # Regularization
    tv_weight: float = 0

    # Decorrelation Params (NEW)
    use_decorrelation: bool = True

    # Gradient Smoothing
    smoothing_sigma_start: float = 0.0
    smoothing_sigma_end: float = 0.0
    smoothing_kernel_size: int = 9

    # Robustness / Augmentation
    jitter_max: int = 2
    rotate_max: float = 1.0
    scale_max: float = 1.05

    # Manifold Constraints (The VAE Safety Guardrails)
    range_weight: float = 0  # Pulls outliers back (Replaces clamping)
    moment_weight: float = 0  # Keeps distribution centered (Mean=0, Std=1)

    # Safe threshold (typically 2.5 to 3.5 standard deviations)
    range_threshold: float = 3.0


# --- Pipeline Setup ---

model_path = "/models/style50"
# model_path = "runwayml/stable-diffusion-v1-5"

try:
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
except:
    print("Could not load local model, loading from HF Hub...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)

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
    def __init__(self, jitter_max=0, rotate_max=0.0, scale_max=1.0):
        self.jitter_max = jitter_max
        self.rotate_max = rotate_max
        self.scale_max = scale_max

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        if self.jitter_max > 0:
            ox = random.randint(-self.jitter_max, self.jitter_max)
            oy = random.randint(-self.jitter_max, self.jitter_max)
            latents = torch.roll(latents, shifts=(ox, oy), dims=(2, 3))

        if self.rotate_max > 0 or self.scale_max != 1.0:
            angle = random.uniform(-self.rotate_max, self.rotate_max)
            scale = random.uniform(1.0, self.scale_max)
            latents = TF.affine(
                latents, angle=angle, translate=[0, 0], scale=scale, shear=0,
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
        self.device = pipe.device
        self.prompt = prompt

    @lru_cache
    def embeddings(self, batch_size: int):
        with torch.no_grad():
            _, negative_embeds = self.pipe.encode_prompt(
                prompt=self.prompt, device=self.device, num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True, negative_prompt=None
            )
        return negative_embeds

    def forward(self, z: torch.Tensor, timestep: int):
        batch_size = z.shape[0]
        embeddings = self.embeddings(batch_size)
        return self.unet(sample=z, timestep=timestep, encoder_hidden_states=embeddings).sample


class EarlyExit(Exception):
    pass


# --- Main Experiment Method ---

def run_experiment(config: ExperimentConfig):
    print(f"\n--- Starting Regularized Optimization ---")

    target_channels = normalize_sequence(config.channels)
    steer_timesteps = normalize_sequence(config.steer_timesteps)
    target_layer = dict(pipe.unet.named_modules())[config.layer_name]
    generator = torch.Generator(device).manual_seed(1024)

    # Pre-calculate FFT Scale Matrix (Reusable)
    if config.use_decorrelation:
        fft_scale = get_preconditioner(64, 64, device)
    else:
        fft_scale = None

    # --- Phase 1: Prior ---
    if config.use_prior:
        print("Phase 1: Generating Steered Prior...")
        steering_fn = create_steering_hook(config.steer_alpha, steer_timesteps, target_channels)
        with steering_context(target_layer, steering_fn):
            output = pipe(config.prompt, num_inference_steps=50, output_type="latent", generator=generator)
        prior_latents = output.images
    else:
        print("Phase 1: Using Gaussian Noise Prior...")
        prior_latents = torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=generator, device=device,
                                    dtype=dtype)

    # --- Phase 2: Optimization ---
    print(f"Phase 2: Optimizing with Decorrelation & Annealing...")

    latents = prior_latents.clone().detach().to(device, dtype=dtype)


    if config.prior_blur_sigma > 0.0:
        latents = gaussian_blur_2d(latents, config.prior_blur_kernel_size, config.prior_blur_sigma)

    image_prior = decode_latents(pipe, latents)
    latents.requires_grad_(True)

    optimizer = torch.optim.Adam([latents], lr=config.learning_rate)
    model_wrapper = UNetWrapper(pipe, config.prompt).eval()
    augmenter = LatentAugmenter(config.jitter_max, config.rotate_max, config.scale_max)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    activations = {}

    def capture_hook(module, input, output):
        activations["act"] = output
        raise EarlyExit

    hook_handle = target_layer.register_forward_hook(capture_hook)

    try:
        pbar = tqdm(range(config.num_steps))
        for step in pbar:
            # Schedule
            progress = step / config.num_steps
            current_t = int(config.start_timestep - (config.start_timestep - config.end_timestep) * progress)
            timestep_tensor = torch.tensor([current_t], device=device, dtype=dtype).int()
            current_sigma = config.smoothing_sigma_start - (
                        config.smoothing_sigma_start - config.smoothing_sigma_end) * progress

            optimizer.zero_grad()
            activations.clear()

            # Augment
            augmented_latents = augmenter(latents)

            if config.see_through_schedule_noise:
                noise = torch.randn_like(augmented_latents)
                z = pipe.scheduler.add_noise(augmented_latents, noise, timestep_tensor)
            else:
                z = augmented_latents

            # Forward
            with autocast(enabled=(device.type == "cuda"), device_type=device.type):
                try:
                    _ = model_wrapper(z, current_t)
                except EarlyExit:
                    pass

                act = activations["act"]
                if isinstance(target_layer, Attention):
                    selected = act[:, :, target_channels]
                elif isinstance(target_layer, Transformer2DModel):
                    selected = act[0][:, target_channels, :, :]
                else:
                    selected = act[:, target_channels, :, :]

                loss_act = -torch.mean(selected)
                loss_tv = config.tv_weight * compute_tv_loss(latents)
                # loss_l2 = config.l2_weight * torch.norm(latents)
                loss_range = config.range_weight * compute_range_loss(latents, threshold=config.range_threshold)

                # B. Moment Loss (Keeps it statistically valid)
                # Ensures the latent 'blob' looks like a normal distribution
                loss_moment = config.moment_weight * compute_moment_loss(latents)

                total_loss = loss_act + loss_tv + loss_range + loss_moment

            # Backward
            if device.type == "cuda":
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                # --- PRECONDITIONING STEP (The New Magic) ---
                if config.use_decorrelation:
                    latents.grad = precondition_gradient(latents.grad, fft_scale)

                # --- GRADIENT SMOOTHING ---
                if current_sigma > 0:
                    latents.grad = gaussian_blur_gradient(latents.grad, config.smoothing_kernel_size, current_sigma)

                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()

                # --- PRECONDITIONING STEP ---
                if config.use_decorrelation:
                    latents.grad = precondition_gradient(latents.grad, fft_scale)

                if current_sigma > 0:
                    latents.grad = gaussian_blur_gradient(latents.grad, config.smoothing_kernel_size, current_sigma)

                optimizer.step()

            pbar.set_description(
                f"t={current_t} | L_act={loss_act.item():.2f} | L_tv={loss_tv.item():.2f} | L_range={loss_range} | L_moment = {loss_moment} | σ={current_sigma:.1f}")

            # show image every 10 steps
            if step % 10 == 0:
                image_dream = decode_latents(pipe, latents)
                show_comparison(image_prior, image_dream, use_prior=config.use_prior)

    finally:
        hook_handle.remove()

    print("Decoding Final Result...")
    image_dream = decode_latents(pipe, latents)
    return image_prior, image_dream


# --- Execution ---

if __name__ == "__main__":
    pipe.scheduler.set_timesteps(50)
    all_timesteps = pipe.scheduler.timesteps.tolist()
    pikes = list(range(0, 400))
    print(f"Piking at {pikes}")
    config = ExperimentConfig(
        layer_name="up_blocks.1.attentions.1",
        channels=1192,
        prompt="",

        # Steering
        use_prior=True, #only for later layers maybe (early layers seem to be doing better without prior, later need it)
        steer_alpha=50.0,
        steer_timesteps=pikes,
        prior_blur_sigma=0.0, #0.5
        prior_blur_kernel_size=25,

        # Optimization
        num_steps=100, # 200 / 100 (for later)
        learning_rate=0.05, # 0.05 for later? we may stick with 0.1 but we need to set the steps lower
        see_through_schedule_noise=True,

        # Annealing
        start_timestep=0, # 0, meaningful pikes and annealing for all meaningful pikes
        end_timestep=0,

        # Regularization
        tv_weight=0.00005, # or 0.0001 or 0.00001

        # --- ENABLE DECORRELATION ---
        use_decorrelation=True, #only for later layers maybe (early layers seem to be doing better without decorelation, later need it)

        # Gradient Smoothing (Can be reduced when decorrelation is on)
        smoothing_sigma_start=0.5,
        smoothing_sigma_end=0.0,

        # Augmentation
        jitter_max=2,
        rotate_max=20.0,
        scale_max=1.05,

        range_weight=1.0,
        moment_weight=0.5,
    )

    prior_img, dreamed_img = run_experiment(config)
    show_comparison(prior_img, dreamed_img, title=f"{config.layer_name}, ch: {config.channels}, tv: {config.tv_weight}",
                    use_prior=config.use_prior)