# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
import os
from datetime import datetime
import getpass
from config import (
    DIG_WEBUI_TITLE,
    DIG_WEBUI_TOP_PAGE_BANNER,
    DIG_WEBUI_FILENAME,
    DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION,
    DIG_PICTURE_NUM_INFERENCE_STEPS,
    DIG_PICTURE_HEIGHT,
    DIG_PICTURE_WIDTH,
    DIG_PICTURE_NEG_PROMPT,
    DIG_PICTURE_GUIDANCE_SCALE
)

# Define fallback prompt if user enters nothing
DEFAULT_PROMPT = "a samurai standing on Mars with a red sun"

torch.cuda.empty_cache()

# Setup save directory
username = getpass.getuser()
save_dir = f"/home/{username}/{DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION}"
os.makedirs(save_dir, exist_ok=True)

# Load the model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    device_map="balanced"
)

pipe.reset_device_map()
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

def generate_image(prompt, user_neg_prompt):
    torch.cuda.empty_cache()

    # Use default values if inputs are empty or only whitespace
    final_prompt = prompt.strip() if prompt and prompt.strip() else DEFAULT_PROMPT
    negative_prompt = user_neg_prompt.strip() if user_neg_prompt and user_neg_prompt.strip() else DIG_PICTURE_NEG_PROMPT

    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=DIG_PICTURE_GUIDANCE_SCALE,
        num_inference_steps=DIG_PICTURE_NUM_INFERENCE_STEPS,
        height=DIG_PICTURE_HEIGHT,
        width=DIG_PICTURE_WIDTH
    ).images[0]

    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{DIG_WEBUI_FILENAME}_{timestamp}.png")
    image.save(filename)

    return image, f"✅ Image saved as {filename}"

# Gradio UI
with gr.Blocks() as image_gen:
    gr.Markdown(f"# {DIG_WEBUI_TOP_PAGE_BANNER}")

    prompt_input = gr.Textbox(
        label="Prompt",
        lines=1,
        placeholder=f"e.g. {DEFAULT_PROMPT}"
    )

    neg_prompt_input = gr.Textbox(
        label="Negative Prompt (optional)",
        lines=1,
        placeholder=f"Default: {DIG_PICTURE_NEG_PROMPT}"
    )

    generate_btn = gr.Button("Generate Image")

    output_image = gr.Image(label="Generated Image")
    output_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, neg_prompt_input],
        outputs=[output_image, output_text]
    )

    prompt_input.submit(
        fn=generate_image,
        inputs=[prompt_input, neg_prompt_input],
        outputs=[output_image, output_text]
    )

image_gen.launch()
