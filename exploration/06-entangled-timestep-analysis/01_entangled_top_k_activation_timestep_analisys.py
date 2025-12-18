import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, Transformer2DModel
from diffusers.models.attention_processor import Attention
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import pickle

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

# --- Configuration ---
layer_name = "up_blocks.1.attentions.1"
TOP_K = 20  # How many 'loudest' channels to consider per timestep

# Define the range you want to iterate through
channels_to_scan = range(1152, 1216)

prompts = [
    "a dog in a style of van gogh",
    "a house in a style of picasso",
    "a tree in a style of cartoon",
    "a car in a style of oil painting"
]

timestep_topk_storage = {}


# --- Helper Function: The Hook ---
def hook(module, input, output):
    t = int(pipe.unet.current_timestep)

    if isinstance(output, tuple):
        out_tensor = output[0]
    elif hasattr(output, "sample"):
        out_tensor = output.sample
    else:
        out_tensor = output

    # Reduce spatial dimensions
    if out_tensor.ndim == 3:  # (Batch, Seq, Channels)
        act_vector = torch.mean(out_tensor, dim=1)
    elif out_tensor.ndim == 4:  # (Batch, Channels, H, W)
        act_vector = torch.mean(out_tensor, dim=[2, 3])
    else:
        act_vector = torch.mean(out_tensor, dim=tuple(range(1, out_tensor.ndim)))

    # Find Top K Indices
    _, top_indices = torch.topk(act_vector, k=TOP_K, dim=1)

    if t not in timestep_topk_storage:
        timestep_topk_storage[t] = []

    timestep_topk_storage[t].extend(top_indices.cpu().numpy())


# --- Load or Run Inference ---
if os.path.exists("timestep_topk_storage.pkl"):
    print("Loading cached data...")
    all_timesteps = pickle.load(open("all_timesteps.pkl", "rb"))
    timestep_topk_storage = pickle.load(open("timestep_topk_storage.pkl", "rb"))
else:
    print("Cache not found. Running inference...")
    model_path = "/models/style50"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    except OSError:
        print("Could not load local model, loading from HF Hub...")
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)

    pipe = pipe.to(device)
    pipe.safety_checker = None

    # Monkey Patching
    original_forward = pipe.unet.forward


    def intercepted_forward(sample, timestep, encoder_hidden_states, **kwargs):
        t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        pipe.unet.current_timestep = t_val
        return original_forward(sample, timestep, encoder_hidden_states, **kwargs)


    pipe.unet.forward = intercepted_forward

    target_layer = dict(pipe.unet.named_modules())[layer_name]
    handle = target_layer.register_forward_hook(hook)

    print(f"Running inference for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        pipe(prompt)
    handle.remove()
    all_timesteps = pipe.scheduler.timesteps.tolist()

    pickle.dump(all_timesteps, open("all_timesteps.pkl", "wb"))
    pickle.dump(timestep_topk_storage, open("timestep_topk_storage.pkl", "wb"))

# --- Processing Loop Over Channels ---

# Create output directory
output_dir = "channel_plots"
os.makedirs(output_dir, exist_ok=True)
print(f"\nStarting analysis for channels {channels_to_scan.start} to {channels_to_scan.stop - 1}...")
print(f"Plots will be saved to ./{output_dir}/")

for channel_id in channels_to_scan:
    counts_per_timestep = []
    valid_timesteps = []

    # 1. Calculate frequencies for THIS specific channel_id
    for t in all_timesteps:
        t_int = int(t)
        if t_int in timestep_topk_storage:
            all_indices_at_t = timestep_topk_storage[t_int]
            hit_count = 0

            # Count occurrences of specific channel_id
            for indices_row in all_indices_at_t:
                # Using numpy sum for speed (True=1, False=0)
                hit_count += np.sum(indices_row == channel_id)

            mean_hits = hit_count / len(prompts)
            counts_per_timestep.append(mean_hits)
            valid_timesteps.append(t)

    # Convert to numpy
    y_observed = np.array(counts_per_timestep)
    x_observed = np.array(valid_timesteps)

    # Optimization: Skip empty channels
    if np.sum(y_observed) == 0:
        continue

    print(f"-> Processing active channel: {channel_id}")

    # 2. Sort & Interpolate
    sort_idx = np.argsort(x_observed)
    x_sorted = x_observed[sort_idx]
    y_sorted = y_observed[sort_idx]

    f_interp = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value=0)
    x_full = np.arange(0, 1001)
    y_full = f_interp(x_full)

    # 3. Find Peaks
    peaks, properties = find_peaks(y_full, height=0.001, distance=20)
    peak_indices = list(peaks)
    peak_heights = list(properties['peak_heights'])

    # Check Edges
    if y_full[0] > 0 and y_full[0] > y_full[1]:
        peak_indices.append(0)
        peak_heights.append(y_full[0])
    if y_full[-1] > 0 and y_full[-1] > y_full[-2]:
        peak_indices.append(1000)
        peak_heights.append(y_full[-1])

    # Sort Peaks
    peak_data = list(zip(peak_indices, peak_heights))
    peak_data.sort(key=lambda x: x[1], reverse=True)
    top_k_peaks = peak_data[:5]

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, y_sorted, 'o', label='Observed', alpha=0.5)
    plt.plot(x_full, y_full, '-', label='Interpolated', alpha=0.8)
    plt.axhline(0, color='black', linewidth=1)

    for t_peak, val_peak in top_k_peaks:
        plt.plot(t_peak, val_peak, 'x', color='red', markersize=10, markeredgewidth=3)
        plt.text(t_peak, val_peak, f" T={t_peak}", verticalalignment='bottom')

    plt.fill_between(x_full, 0, y_full, alpha=0.2, color='blue')
    plt.gca().invert_xaxis()
    plt.xlabel("Timestep")
    plt.ylabel(f"Freq in Top-{TOP_K}")
    plt.title(f"Channel {channel_id} Frequency Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save and Close
    plt.savefig(f"{output_dir}/channel_{channel_id}.png")
    plt.close()  # Important to free memory!

print("Done.")