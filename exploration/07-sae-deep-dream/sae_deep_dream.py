from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Union, Sequence, Tuple, NamedTuple, Optional
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import math
import numpy as np
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
import einops

# Diffusers imports
from diffusers import StableDiffusionPipeline
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.attention_processor import Attention
from torch.amp import GradScaler, autocast
import torchvision.transforms.functional as TF

# --- Setup Device ---
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


# -------------------------------------------------------------------------
# --- SAE DEFINITIONS ---
# -------------------------------------------------------------------------

@dataclass
class SaeConfig:
    expansion_factor: int = 32
    normalize_decoder: bool = True
    num_latents: int = 0
    k: int = 32
    batch_topk: bool = False
    sample_topk: bool = False
    input_unit_norm: bool = False
    multi_topk: bool = False

    @classmethod
    def from_dict(cls, d, drop_extra_fields=True):
        valid_keys = cls.__annotations__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self):
        return self.__dict__


class Sae(nn.Module):
    def __init__(self, d_in: int, cfg: SaeConfig, device="cpu", dtype=None, decoder: bool = True):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = (
            nn.Parameter(self.encoder.weight.data[:, :d_in].clone())
            if decoder else None
        )
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_from_hub(name: str, hookpoint: str | None = None, device="cpu", decoder: bool = True) -> "Sae":
        repo_path = Path(snapshot_download(name, allow_patterns=f"{hookpoint}/*" if hookpoint else None))
        if hookpoint:
            repo_path = repo_path / hookpoint
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")
        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(path: Path | str, device="cpu", decoder: bool = True) -> "Sae":
        path = Path(path)
        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(model=sae, filename=str(path / "sae.safetensors"), device=str(device), strict=decoder)
        return sae

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        # --- ROBUST MPS FIX ---
        target_dtype = self.encoder.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        sae_in = x - self.b_dec

        # Ensure Contiguity (Vital for MPS MatMul)
        if not sae_in.is_contiguous():
            sae_in = sae_in.contiguous()

        out = self.encoder(sae_in)
        return nn.functional.relu(out)

    def preprocess_input(self, x):
        # x: [B, H, W, C] or [B*H*W, C]
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C

        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])

        # MPS Fix
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        if self.cfg.input_unit_norm:
            x_mean = x_flat.mean(dim=-1, keepdim=True)
            x_flat = x_flat - x_mean
            x_std = x_flat.std(dim=-1, keepdim=True)
            x_flat = x_flat / (x_std + 1e-5)
            return x_flat, x_mean, x_std, original_shape
        else:
            return x_flat, None, None, original_shape

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        if self.W_dec is None: return
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps


# -------------------------------------------------------------------------
# --- HELPERS ---
# -------------------------------------------------------------------------

def normalize_sequence(input_val: Union[int, Sequence[int], range]) -> List[int]:
    if isinstance(input_val, int): return [input_val]
    if isinstance(input_val, range): return list(input_val)
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
    plt.title("Phase 1: Prior (Steered)" if use_prior else "Phase 1: Noise")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_dreamed)
    plt.title("Phase 2: SAE Deep Dream")
    plt.axis('off')
    plt.show()


# --- Regularization Utilities ---
def compute_tv_loss(x: torch.Tensor) -> torch.Tensor:
    h_x, w_x = x.size(2), x.size(3)
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return h_tv + w_tv


def gaussian_blur_gradient(grad: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if sigma <= 0: return grad
    if kernel_size % 2 == 0: kernel_size += 1
    return TF.gaussian_blur(grad, kernel_size=kernel_size, sigma=sigma)


def get_preconditioner(h, w, device):
    Y, X = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    X, Y = X - w // 2, Y - h // 2
    freq_sq = X ** 2 + Y ** 2
    freqs = torch.sqrt(freq_sq.float())
    scale = 1.0 / (freqs + 1.0)
    return torch.fft.ifftshift(scale)


def precondition_gradient(grad: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
    grad_fft = torch.fft.fft2(grad.float())
    grad_fft_scaled = grad_fft * scale_matrix.unsqueeze(0).unsqueeze(0)
    grad_decorrelated = torch.fft.ifft2(grad_fft_scaled).real
    norm_factor = grad.std() / (grad_decorrelated.std() + 1e-8)
    return grad_decorrelated.to(grad.dtype) * norm_factor


def gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)


def compute_range_loss(latents: torch.Tensor, threshold: float = 2.5) -> torch.Tensor:
    magnitude = torch.abs(latents)
    out_of_bounds = F.relu(magnitude - threshold)
    return torch.mean(out_of_bounds ** 2)


def compute_moment_loss(latents: torch.Tensor) -> torch.Tensor:
    return torch.norm(latents.mean()) + torch.norm(latents.std() - 1.0)


# -------------------------------------------------------------------------
# --- EXPERIMENT CONFIG ---
# -------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # SAE Config
    sae_hub_path: str

    # MODIFIED: Accepts single int OR list of ints
    sae_feature_indices: Union[int, List[int]]
    layer_name: str

    prompt: str = ""

    # Phase 1: Steering Params
    use_prior: bool = True
    steer_strength: float = 10.0
    steer_timesteps: Union[int, Sequence[int]] = range(0, 1000)
    prior_blur_sigma: float = 0.0
    prior_blur_kernel_size: int = 9

    # Phase 2: Deep Dream Optimization Params
    num_steps: int = 500
    learning_rate: float = 0.05
    see_through_schedule_noise: bool = True

    # Weight for suppressing OTHER features
    suppress_weight: float = 0.0

    # Annealing Schedule
    start_timestep: int = 600
    end_timestep: int = 100

    # Regularization
    tv_weight: float = 0
    use_decorrelation: bool = True
    smoothing_sigma_start: float = 0.0
    smoothing_sigma_end: float = 0.0
    smoothing_kernel_size: int = 9

    # Robustness / Augmentation
    jitter_max: int = 2
    rotate_max: float = 1.0
    scale_max: float = 1.05

    # Manifold Constraints
    range_weight: float = 0
    moment_weight: float = 0
    range_threshold: float = 3.0


# -------------------------------------------------------------------------
# --- PIPELINE & HOOKS ---
# -------------------------------------------------------------------------

model_path = "../../models/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe = pipe.to(device)
pipe.safety_checker = None

original_forward = pipe.unet.forward


def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
    t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    pipe.unet.current_timestep = t_val
    return original_forward(sample, timestep, encoder_hidden_states, **kwargs)


pipe.unet.forward = intercepted_forward


class LatentAugmenter:
    def __init__(self, jitter_max=0, rotate_max=0.0, scale_max=1.0):
        self.jitter_max = jitter_max
        self.rotate_max = rotate_max
        self.scale_max = scale_max

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        if self.jitter_max > 0:
            ox, oy = random.randint(-self.jitter_max, self.jitter_max), random.randint(-self.jitter_max,
                                                                                       self.jitter_max)
            latents = torch.roll(latents, shifts=(ox, oy), dims=(2, 3))

        if self.rotate_max > 0 or self.scale_max != 1.0:
            angle = random.uniform(-self.rotate_max, self.rotate_max)
            scale = random.uniform(1.0, self.scale_max)
            latents = TF.affine(latents, angle=angle, translate=[0, 0], scale=scale, shear=0,
                                interpolation=TF.InterpolationMode.BILINEAR)
        return latents


@contextmanager
def steering_context(module: nn.Module, hook_fn):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield handle
    finally:
        handle.remove()


# --- SAE Steering Hook (Modified for Multiple Channels) ---
class SAESteeringHook:
    def __init__(self, sae: Sae, feature_indices: List[int], strength: float, target_timesteps: List[int]):
        self.sae = sae
        self.feature_indices = feature_indices
        self.strength = strength
        self.target_timesteps = set(target_timesteps)

    def __call__(self, module, input, output):
        t = getattr(pipe.unet, 'current_timestep', None)

        if isinstance(output, tuple):
            h_states = output[0]
            is_tuple = True
        else:
            h_states = output
            is_tuple = False

        if t is not None and t in self.target_timesteps:
            # 1. Preprocess
            x_flat, x_mean, x_std, original_shape = self.sae.preprocess_input(h_states)
            # 2. Encode
            pre_acts = self.sae.pre_acts(x_flat)

            # 3. INTERVENTION (Iterate over all targets)
            for idx in self.feature_indices:
                pre_acts[:, idx] = self.strength

            # 4. Activation
            acts = torch.nn.functional.relu(pre_acts)
            # 5. Decode
            recon = acts.to(self.sae.W_dec.dtype) @ self.sae.W_dec + self.sae.b_dec

            if self.sae.cfg.input_unit_norm and x_mean is not None:
                recon = recon * x_std + x_mean

            # 6. Reshape back
            recon = recon.reshape(original_shape).permute(0, 3, 1, 2)
            recon = recon.to(h_states.dtype)

            if is_tuple:
                return (recon,) + output[1:]
            return recon

        return output


class UNetWrapper(nn.Module):
    def __init__(self, pipe, prompt):
        super().__init__()
        self.pipe = pipe
        self.unet = pipe.unet
        self.device = pipe.device
        self.prompt = prompt

    @lru_cache
    def embeddings(self, batch_size):
        with torch.no_grad():
            _, negative_embeds = self.pipe.encode_prompt(
                prompt=self.prompt, device=self.device, num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True, negative_prompt=None
            )
        return negative_embeds

    def forward(self, z, timestep):
        embeddings = self.embeddings(z.shape[0])
        return self.unet(sample=z, timestep=timestep, encoder_hidden_states=embeddings).sample


class EarlyExit(Exception): pass


# -------------------------------------------------------------------------
# --- EXPERIMENT RUNNER ---
# -------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig):
    print(f"\n--- Starting SAE Deep Dream ---")

    # 1. Load SAE
    print(f"Loading SAE from {config.sae_hub_path}...")
    try:
        sae = Sae.load_from_hub(
            config.sae_hub_path,
            hookpoint=config.layer_name,
            device=device
        ).to(dtype)
    except Exception as e:
        print(f"Error loading SAE: {e}. Ensure the Hub Path and Hookpoint are correct.")
        return None, None

    # Normalize targets to a list
    if isinstance(config.sae_feature_indices, int):
        target_indices = [config.sae_feature_indices]
    else:
        target_indices = list(config.sae_feature_indices)

    print(f"Target Features: {target_indices}")

    # Robust layer selection
    layer_key = config.layer_name
    if layer_key.startswith("unet."):
        layer_key = layer_key.replace("unet.", "", 1)

    target_layer = dict(pipe.unet.named_modules())[layer_key]
    steer_timesteps = normalize_sequence(config.steer_timesteps)
    generator = torch.Generator(device).manual_seed(1024)

    fft_scale = get_preconditioner(64, 64, device) if config.use_decorrelation else None

    # --- Phase 1: Steering (Prior Generation) ---
    if config.use_prior:
        print("Phase 1: Generating Steered Prior using SAE Intervention...")
        hook_obj = SAESteeringHook(
            sae=sae,
            feature_indices=target_indices,
            strength=config.steer_strength,
            target_timesteps=steer_timesteps
        )
        with steering_context(target_layer, hook_obj):
            output = pipe(config.prompt, num_inference_steps=50, output_type="latent", generator=generator)
        prior_latents = output.images
    else:
        print("Phase 1: Using Gaussian Noise Prior...")
        prior_latents = torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=generator, device=device,
                                    dtype=dtype)

    # --- Phase 2: Optimization ---
    print(f"Phase 2: Optimizing SAE Features {target_indices}...")

    latents = prior_latents.clone().detach().to(device, dtype=dtype)
    if config.prior_blur_sigma > 0.0:
        latents = gaussian_blur_2d(latents, config.prior_blur_kernel_size, config.prior_blur_sigma)

    image_prior = decode_latents(pipe, latents)
    latents.requires_grad_(True)

    optimizer = torch.optim.AdamW([latents], lr=config.learning_rate)
    model_wrapper = UNetWrapper(pipe, config.prompt).eval()
    augmenter = LatentAugmenter(config.jitter_max, config.rotate_max, config.scale_max)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    activations = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            activations["act"] = output[0]
        else:
            activations["act"] = output
        raise EarlyExit

    hook_handle = target_layer.register_forward_hook(capture_hook)

    try:
        pbar = tqdm(range(config.num_steps))
        for step in pbar:
            progress = step / config.num_steps
            current_t = int(config.start_timestep - (config.start_timestep - config.end_timestep) * progress)
            timestep_tensor = torch.tensor([current_t], device=device, dtype=dtype).int()
            current_sigma = config.smoothing_sigma_start - (
                    config.smoothing_sigma_start - config.smoothing_sigma_end) * progress

            optimizer.zero_grad()
            activations.clear()

            # Augment & Noise
            augmented_latents = augmenter(latents)
            if config.see_through_schedule_noise:
                noise = torch.randn_like(augmented_latents)
                z = pipe.scheduler.add_noise(augmented_latents, noise, timestep_tensor)
            else:
                z = augmented_latents

            # Forward UNet
            with autocast(enabled=(device.type == "cuda"), device_type=device.type):
                try:
                    _ = model_wrapper(z, current_t)
                except EarlyExit:
                    pass

                unet_act = activations["act"]

                # --- SAE LOSS CALCULATION ---
                # 1. Reshape
                x_flat, _, _, _ = sae.preprocess_input(unet_act)

                # 2. Encode
                pre_acts = sae.pre_acts(x_flat)
                sae_feature_acts = F.relu(pre_acts)  # [Batch, Num_Features]

                # 3. Select activations for ALL target indices
                # We can simple sum the activations of all targets
                target_acts = torch.stack([sae_feature_acts[:, idx] for idx in target_indices], dim=1)

                # Mean across batch, then Sum across selected features
                # Or Mean across everything. Summing ensures more features = stronger gradient.
                loss_target = -torch.mean(torch.sum(target_acts, dim=1))

                # Suppression Logic
                # Sum of ALL activations
                sum_all = torch.sum(sae_feature_acts, dim=1)

                # Sum of TARGETS
                sum_targets = torch.sum(target_acts, dim=1)

                # Sum of OTHERS
                sum_others = sum_all - sum_targets

                loss_suppress = torch.mean(sum_others)

                loss_sae = loss_target + (config.suppress_weight * loss_suppress)

                # Regularization
                loss_tv = config.tv_weight * compute_tv_loss(latents)
                loss_range = config.range_weight * compute_range_loss(latents, threshold=config.range_threshold)
                loss_moment = config.moment_weight * compute_moment_loss(latents)

                total_loss = loss_sae + loss_tv + loss_range + loss_moment

            # Backward & Step
            if device.type == "cuda":
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if config.use_decorrelation:
                    latents.grad = precondition_gradient(latents.grad, fft_scale)
                if current_sigma > 0:
                    latents.grad = gaussian_blur_gradient(latents.grad, config.smoothing_kernel_size, current_sigma)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config.use_decorrelation:
                    latents.grad = precondition_gradient(latents.grad, fft_scale)
                if current_sigma > 0:
                    latents.grad = gaussian_blur_gradient(latents.grad, config.smoothing_kernel_size, current_sigma)
                optimizer.step()

            pbar.set_description(
                f"t={current_t} | L_sae={loss_sae.item():.2f} | L_target={loss_target} | L_suppress={loss_suppress} | L_tv={loss_tv.item():.2f} | L_range={loss_range} | L_moment = {loss_moment} | σ={current_sigma:.1f}")

            if step % 10 == 0:
                img_live = decode_latents(pipe, latents)

                # plot
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(img_live)
                plt.subplot(1, 2, 2)
                plt.imshow(image_prior)
                plt.show()
    finally:
        hook_handle.remove()

    print("Decoding Final Result...")
    image_dream = decode_latents(pipe, latents)
    return image_prior, image_dream


# -------------------------------------------------------------------------
# --- MAIN ---
# -------------------------------------------------------------------------

if __name__ == "__main__":
    pipe.scheduler.set_timesteps(50)
    timesteps = pipe.scheduler.timesteps
    timestep_10 = timesteps[-10]

    config = ExperimentConfig(
        # SAE Params
        sae_hub_path="bcywinski/SAeUron",
        layer_name="unet.up_blocks.1.attentions.1",

        # --- MODIFIED: List of features ---
        sae_feature_indices=[19324],

        prompt="",

        # Steering
        use_prior=True,
        steer_strength=50.0,
        steer_timesteps=range(0, 600),
        prior_blur_sigma=0.0, # to delete
        prior_blur_kernel_size=25, # to delete

        # Optimization
        num_steps=2000,
        learning_rate=0.1,
        see_through_schedule_noise=True,

        suppress_weight=0.005,

        start_timestep=0, # to delete
        end_timestep=0, # to delete

        # Regularization
        tv_weight=0.00005,
        use_decorrelation=True,
        smoothing_sigma_start=0.5,
        smoothing_sigma_end=0.0,

        # Augmentation
        jitter_max=4,
        rotate_max=20.0,
        scale_max=1.5,

        range_weight=1.0,
        moment_weight=0.5,
    )

    prior, result = run_experiment(config)

    if prior is not None:
        show_comparison(prior, result,
                        title=f"SAE Features: {config.sae_feature_indices} @ {config.layer_name}",
                        use_prior=config.use_prior
                        )