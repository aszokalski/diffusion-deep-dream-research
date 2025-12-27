from diffusion_deep_dream_research.model.hooked_model_wrapper import HookedModelWrapper
from diffusers import StableDiffusionPipeline
import torch

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

model_path = "../models/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe = pipe.to(device)

wrapper = HookedModelWrapper.from_layer(
    pipe=pipe,
    target_layer_name="up_blocks.1.attentions.1"
)

res=wrapper.forward_with_capture(
    prompts=[
        "a dog in a style of van gogh",
        "a house in a style of picasso",
    ],
    num_images_per_prompt=2
)

activations = res.hook_activations
images = res.images

print(f"Captured activations at timesteps: {list(activations.keys())}")
for t, act in activations.items():
    print(f"Timestep {t}: activation shape {act.shape}")
    print(f"Timestep {t}: activation min {act.min()}, max {act.max()}")

import matplotlib.pyplot as plt

for i, img in enumerate(images):
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Generated Image {i+1}")
    plt.show()