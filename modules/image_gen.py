# Written by StormTheory
# https://github.com/stormtheory/friday-ai

#from transformers import CLIPImageProcessor
import os
import subprocess
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
import getpass
from config import IMAGE_GEN_IMAGE_SAVE_HOMESPACE_LOCATION

username = getpass.getuser()

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

def generate_image(prompt: str, output_dir=f"/home/{username}/{IMAGE_GEN_IMAGE_SAVE_HOMESPACE_LOCATION}"):
    # Create images/ directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    output_path = os.path.join(output_dir, filename)

    print(f"🧠 Generating image from prompt: '{prompt}'")
    result = pipe(prompt)
    image = result.images[0]
    image.save(output_path)
    
    abs_path = os.path.abspath(output_path)
    print(f"✅ Image saved: {abs_path}")

    try:
        subprocess.run(["xdg-open", abs_path])
    except Exception as e:
        print(f"⚠️ Could not open image: {e}")

    return abs_path
