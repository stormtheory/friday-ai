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

# Default prompt if none is provided
DEFAULT_PROMPT = "a samurai standing on Mars with a red sun"

torch.cuda.empty_cache()

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

def generate_image(prompt, user_neg_prompt, user_guidance_scale, user_steps, user_width, user_height, user_filename_prefix, user_save_location):
    torch.cuda.empty_cache()

    username = getpass.getuser()

    final_prompt = prompt.strip() if prompt and prompt.strip() else DEFAULT_PROMPT
    negative_prompt = user_neg_prompt.strip() if user_neg_prompt and user_neg_prompt.strip() else DIG_PICTURE_NEG_PROMPT
    guidance_scale = float(user_guidance_scale) if user_guidance_scale else DIG_PICTURE_GUIDANCE_SCALE
    steps = int(user_steps) if user_steps else DIG_PICTURE_NUM_INFERENCE_STEPS
    width = int(user_width) if user_width else DIG_PICTURE_WIDTH
    height = int(user_height) if user_height else DIG_PICTURE_HEIGHT
    filename_prefix = user_filename_prefix.strip() if user_filename_prefix and user_filename_prefix.strip() else DIG_WEBUI_FILENAME
    save_subpath = user_save_location.strip() if user_save_location and user_save_location.strip() else DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION

    # Final save directory
    save_dir = f"/home/{username}/{save_subpath}"
    os.makedirs(save_dir, exist_ok=True)

    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=height,
        width=width
    ).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.png")
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

    with gr.Row(scale=0.5):
        guidance_input = gr.Number(
                label="Guidance Scale [1 - 20]",
                minimum=1.0,
                maximum=20.0,
                value=None,
                placeholder=f"{DIG_PICTURE_GUIDANCE_SCALE}"
            )
        
        steps_input = gr.Number(
                label="Inference Steps [20 - 100]",
                minimum=20,
                maximum=100,
                value=None,
                placeholder=f"{DIG_PICTURE_NUM_INFERENCE_STEPS}"
            )
        
        width_input = gr.Number(
                label="Width",
                minimum=64,
                maximum=2048,
                value=None,
                placeholder=f"{DIG_PICTURE_WIDTH}"
            )
        
        height_input = gr.Number(
                label="Height",
                minimum=64,
                maximum=2048,
                value=None,
                placeholder=f"{DIG_PICTURE_HEIGHT}"
            )

    
    with gr.Row(scale=0.5):
        save_location_input = gr.Textbox(
            label="Image Save Subfolder within User Home",
            placeholder=f"{DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION}"
        )
        filename_prefix_input = gr.Textbox(
                label="Filename Prefix",
                placeholder=f"{DIG_WEBUI_FILENAME}"
            )

    generate_btn = gr.Button("Generate Image")

    output_image = gr.Image(label="Generated Image")
    output_text = gr.Textbox(label="Status", interactive=False)

    inputs_list = [
        prompt_input,
        neg_prompt_input,
        guidance_input,
        steps_input,
        width_input,
        height_input,
        filename_prefix_input,
        save_location_input
    ]

    generate_btn.click(
        fn=generate_image,
        inputs=inputs_list,
        outputs=[output_image, output_text]
    )

    prompt_input.submit(
        fn=generate_image,
        inputs=inputs_list,
        outputs=[output_image, output_text]
    )

image_gen.launch()
