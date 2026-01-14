import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, Transformer2DModel
from diffusers.models.attention_processor import Attention
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

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

# --- Pipeline Setup ---
model_path = "/models/style50"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
except OSError:
    print("Could not load local model, loading from HF Hub...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)

pipe = pipe.to(device)
pipe.safety_checker = None

# --- Monkey Patching ---
original_forward = pipe.unet.forward


def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
    t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
    pipe.unet.current_timestep = t_val
    return original_forward(sample, timestep, encoder_hidden_states, **kwargs)


pipe.unet.forward = intercepted_forward

# --- Configuration ---
layer_name = "up_blocks.1.attentions.1"
target_layer = dict(pipe.unet.named_modules())[layer_name]
channels = range(1152, 1216)
prompts = [
    "a dog in a style of van gogh",
    "a house in a style of picasso",
    "a tree in a style of cartoon",
    "a car in a style of oil painting"
]

timestep_accumulator = {}


def hook(module, input, output):
    t = int(pipe.unet.current_timestep)
    if isinstance(module, Attention):
        val = torch.mean(output[:, :, channels]).float()
    elif isinstance(module, Transformer2DModel):
        val = torch.mean(output[0][:, channels, :, :]).float()
    else:
        val = torch.mean(output[:, channels, :, :]).float()

    val = val.cpu().detach().item()
    if t not in timestep_accumulator:
        timestep_accumulator[t] = []
    timestep_accumulator[t].append(val)


handle = target_layer.register_forward_hook(hook)

# --- Run Inference ---
print(f"Running inference for {len(prompts)} prompts...")
for i, prompt in enumerate(prompts):
    pipe(prompt)
handle.remove()

# --- Process Data ---
all_timesteps = pipe.scheduler.timesteps.tolist()
avg_acts = []
valid_timesteps = []

for t in all_timesteps:
    t_int = int(t)
    if t_int in timestep_accumulator:
        values = timestep_accumulator[t_int]
        avg_acts.append(np.mean(values))
        valid_timesteps.append(t)

# Convert to numpy
x_observed = np.array(valid_timesteps)
y_observed = np.array(avg_acts)

# 1. SORT DATA for Interpolation (numpy.interp requires increasing x)
sort_idx = np.argsort(x_observed)
x_sorted = x_observed[sort_idx]
y_sorted = y_observed[sort_idx]

# 2. INTERPOLATE to get full 0-1000 range
# 'linear' is safer than 'cubic' for noisy activation data to avoid overshooting
f_interp = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value="extrapolate")

x_full = np.arange(0, 1001)  # 0 to 1000
y_full = f_interp(x_full)
global_median = np.median(y_full)

# --- GOAL 1: List of all timesteps > Median ---
timesteps_above_median = np.where(y_full > global_median)[0]

# --- GOAL 2: Find Peaks (Including Edges) ---
# Step A: Find internal peaks
peaks, properties = find_peaks(y_full, height=global_median, distance=20)
peak_indices = list(peaks)
peak_heights = list(properties['peak_heights'])

# Step B: Explicitly check edges (0 and 1000) for "monotonic rise" behavior
# Check Start (0): If 0 > 1 (descending from 0) and 0 > median, it's a peak.
if y_full[0] > global_median and y_full[0] > y_full[1]:
    peak_indices.append(0)
    peak_heights.append(y_full[0])

# Check End (1000): If 1000 > 999 (ascending to 1000) and 1000 > median, it's a peak.
if y_full[-1] > global_median and y_full[-1] > y_full[-2]:
    peak_indices.append(1000)
    peak_heights.append(y_full[-1])

# Step C: Combine, Sort, and Filter Top K
peak_data = list(zip(peak_indices, peak_heights))
# Sort descending by height (x[1])
peak_data.sort(key=lambda x: x[1], reverse=True)# Select top K
K = 5
top_k_peaks = peak_data[:K]

# --- Printing Results ---
print("-" * 30)
print(f"Global Median Activation (0-1000): {global_median:.4f}")
print("-" * 30)

print(f"1. Timesteps where activation > median (Total: {len(timesteps_above_median)})")
# Printing just a summary so we don't spam the console
print(f"   Range: {timesteps_above_median}")

print("-" * 30)
print(f"2. Top {K} Peaks (Timestep, Activation):")
for t_peak, act_peak in top_k_peaks:
    print(f"   T = {t_peak}: {act_peak:.4f}")
print("-" * 30)

# --- Plotting Verification ---
plt.figure(figsize=(10, 6))
plt.plot(x_sorted, y_sorted, 'o', label='Observed (Scheduler)', alpha=0.5)
plt.plot(x_full, y_full, '-', label='Interpolated (0-1000)', alpha=0.5)
plt.axhline(global_median, color='red', linestyle='--', label='Median')

# Plot peaks
for t_peak, act_peak in top_k_peaks:
    plt.plot(t_peak, act_peak, 'x', color='black', markersize=10, markeredgewidth=3)
    plt.text(t_peak, act_peak, f" T={t_peak}", verticalalignment='bottom')

# Highlight the range above median
plt.fill_between(x_full, global_median, y_full, where=(y_full > global_median),
                 color='green', alpha=0.1, label='> Median')

plt.gca().invert_xaxis()  # Diffusion goes 1000 -> 0
plt.xlabel("Timestep")
plt.ylabel("Activation")
plt.title(f"Interpolation & Peak Detection (Top {K})")
plt.legend()
plt.show()

