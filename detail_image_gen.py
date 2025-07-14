# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import gradio as gr
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
import torch
import os
from datetime import datetime
import getpass
import torch

torch.cuda.empty_cache()

# Setup save directory
username = getpass.getuser()
save_dir = f"/home/{username}/Pictures/AI"
os.makedirs(save_dir, exist_ok=True)

# Load SDXL model (1 time only)
#pipe = StableDiffusionXLPipeline.from_pretrained(
#    "stabilityai/stable-diffusion-xl-base-1.0",
#    torch_dtype=torch.float16,
#    variant="fp16",
#    use_safetensors=True
#).to("cuda")

# Load the model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    device_map="balanced"  # Important: Auto-offloads to CPU if needed
)

pipe.reset_device_map()
pipe.enable_model_cpu_offload()  # 👈 Try offloading parts of model to CPU RAM

# Optional: Reduce VRAM usage further
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()  # Reduces memory use

def generate_image(prompt):
    torch.cuda.empty_cache()

    if not prompt or prompt.strip() == "":
        return None, "beautiful tigers."

    negative_prompt = "blurry, low quality, distorted, artifacts, text, watermark"

    # Generate image with memory-friendly settings
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        #num_inference_steps=20,
        num_inference_steps=25,
        #height=512,
        #width=768
        height=1024,
        width=1024
    ).images[0]

    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"detailed_XL_friday_{timestamp}.png")
    image.save(filename)

    return image, f"✅ Image saved as {filename}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Stable Diffusion Image Generator - F.R.I.D.A.Y.")

    prompt_input = gr.Textbox(label="Enter prompt for image generation", lines=1, placeholder="e.g. a samurai standing on Mars with a red sun")

    generate_btn = gr.Button("Generate Image")

    output_image = gr.Image(label="Generated Image")
    output_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(fn=generate_image, inputs=prompt_input, outputs=[output_image, output_text])
    prompt_input.submit(fn=generate_image, inputs=prompt_input, outputs=[output_image, output_text])

demo.launch()
