# Written by StormTheory
# https://github.com/stormtheory/friday-ai

import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
import os
from datetime import datetime
import getpass
import json
import re
from config import DIG_DEFAULT_PROMPT as DEFAULT_PROMPT
from config import (
    DIG_WEBUI_TITLE,
    DIG_WEBUI_TOP_PAGE_BANNER,
    DIG_WEBUI_FILENAME,
    DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION,
    DIG_PICTURE_NUM_INFERENCE_STEPS,
    DIG_PICTURE_HEIGHT,
    DIG_PICTURE_WIDTH,
    DIG_PICTURE_NEG_PROMPT,
    DIG_PICTURE_GUIDANCE_SCALE,
    DIG_WEBUI_THREAD_DATA_DIR
)


# 🔐 Thread storage directory
DEFAULT_THREAD_NAME = "Default"
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

os.makedirs(DIG_WEBUI_THREAD_DATA_DIR, exist_ok=True)

# Default config for threads
DEFAULT_CONFIG = {
    "prompt": DEFAULT_PROMPT,
    "neg_prompt": DIG_PICTURE_NEG_PROMPT,
    "guidance_scale": DIG_PICTURE_GUIDANCE_SCALE,
    "steps": DIG_PICTURE_NUM_INFERENCE_STEPS,
    "width": DIG_PICTURE_WIDTH,
    "height": DIG_PICTURE_HEIGHT,
    "filename_prefix": DIG_WEBUI_FILENAME,
    "save_location": DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION
}

# Save default thread if not exists
default_config_path = os.path.join(DIG_WEBUI_THREAD_DATA_DIR, DEFAULT_THREAD_NAME + ".json")
if not os.path.exists(default_config_path):
    with open(default_config_path, "w") as f:
        json.dump(DEFAULT_CONFIG, f)

# --- Utility Functions ---
def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def save_config(name, config):
    path = os.path.join(DIG_WEBUI_THREAD_DATA_DIR, sanitize_filename(name) + ".json")
    with open(path, "w") as f:
        json.dump(config, f)

def load_config(name):
    path = os.path.join(DIG_WEBUI_THREAD_DATA_DIR, sanitize_filename(name) + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_CONFIG

def list_threads():
    threads = [f[:-5] for f in os.listdir(DIG_WEBUI_THREAD_DATA_DIR) if f.endswith(".json")]
    if DEFAULT_THREAD_NAME not in threads:
        threads.insert(0, DEFAULT_THREAD_NAME)
    else:
        threads = sorted(set([DEFAULT_THREAD_NAME] + threads))
    return threads

def load_threads():
    return gr.update(choices=list_threads(), value=DEFAULT_THREAD_NAME)

def delete_thread(name):
    if name == DEFAULT_THREAD_NAME:
        return False
    path = os.path.join(DIG_WEBUI_THREAD_DATA_DIR, sanitize_filename(name) + ".json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def handle_update_thread(selected_name, prompt, neg, gs, steps, w, h, fn_prefix, save_loc):
    selected_name = selected_name.strip()
    if selected_name == DEFAULT_THREAD_NAME:
        return "⚠️ Cannot update the Default thread."

    config = {
        "prompt": prompt,
        "neg_prompt": neg,
        "guidance_scale": gs if gs is not None else DIG_PICTURE_GUIDANCE_SCALE,
        "steps": steps if steps is not None else DIG_PICTURE_NUM_INFERENCE_STEPS,
        "width": w if w is not None else DIG_PICTURE_WIDTH,
        "height": h if h is not None else DIG_PICTURE_HEIGHT,
        "filename_prefix": fn_prefix,
        "save_location": save_loc
    }

    save_config(selected_name, config)
    return f"✅ Updated thread '{selected_name}'"


# --- Thread Handlers ---
def handle_save_thread(name, prompt, neg, gs, steps, w, h, fn_prefix, save_loc):
    name = name.strip()
    if not name:
        return gr.update(visible=True), "⚠️ Please enter a thread name.", gr.update()
    if name == DEFAULT_THREAD_NAME:
        return gr.update(visible=True), "⚠️ Cannot overwrite the Default thread.", gr.update()

    config = {
        "prompt": prompt,
        "neg_prompt": neg,
        "guidance_scale": gs if gs is not None else DIG_PICTURE_GUIDANCE_SCALE,
        "steps": steps if steps is not None else DIG_PICTURE_NUM_INFERENCE_STEPS,
        "width": w if w is not None else DIG_PICTURE_WIDTH,
        "height": h if h is not None else DIG_PICTURE_HEIGHT,
        "filename_prefix": fn_prefix,
        "save_location": save_loc
    }

    save_config(name, config)
    return (
        gr.update(choices=list_threads(), value=name),
        f"✅ Saved thread '{name}'",
        gr.update(visible=False)
    )

def handle_load_thread(name):
    cfg = load_config(name)
    return (
        cfg["prompt"],
        cfg["neg_prompt"],
        cfg["guidance_scale"],
        cfg["steps"],
        cfg["width"],
        cfg["height"],
        cfg["filename_prefix"],
        cfg["save_location"]
    )

def handle_delete_thread(name):
    if delete_thread(name):
        updated = list_threads()
        return gr.update(choices=updated, value=DEFAULT_THREAD_NAME), f"🗑️ Deleted thread '{name}'"
    return gr.update(), "⚠️ Cannot delete this thread."

# --- Core Image Generation ---
def generate_image(prompt, user_neg_prompt, user_guidance_scale, user_steps, user_width, user_height, user_filename_prefix, user_save_location):
    torch.cuda.empty_cache()
    username = getpass.getuser()

    final_prompt = prompt.strip() if prompt and prompt.strip() else DEFAULT_PROMPT
    negative_prompt = user_neg_prompt.strip() if user_neg_prompt and user_neg_prompt.strip() else DIG_PICTURE_NEG_PROMPT
    guidance_scale = float(user_guidance_scale) if user_guidance_scale is not None else DIG_PICTURE_GUIDANCE_SCALE
    steps = int(user_steps) if user_steps is not None else DIG_PICTURE_NUM_INFERENCE_STEPS
    width = int(user_width) if user_width is not None else DIG_PICTURE_WIDTH
    height = int(user_height) if user_height is not None else DIG_PICTURE_HEIGHT
    filename_prefix = user_filename_prefix.strip() if user_filename_prefix and user_filename_prefix.strip() else DIG_WEBUI_FILENAME
    save_subpath = user_save_location.strip() if user_save_location and user_save_location.strip() else DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION

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
    # 🧼 Clean image (strip EXIF + auto-orient and Meta Data for safety & display consistency)
    from PIL import ImageOps
    clean_image = ImageOps.exif_transpose(image).convert("RGB")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.png")
    clean_image.save(filename)

    return image, f"✅ Image saved as {filename}"

# --- Gradio UI ---
with gr.Blocks() as image_gen:
    gr.Markdown(f"# {DIG_WEBUI_TOP_PAGE_BANNER}")

    prompt_input = gr.Textbox(label="Prompt", lines=1, placeholder=f"e.g. {DEFAULT_PROMPT}")

    with gr.Accordion("Advanced Settings", open=False):
        neg_prompt_input = gr.Textbox(label="Negative Prompt", lines=1, value=DIG_PICTURE_NEG_PROMPT)
        with gr.Row():
            guidance_input = gr.Number(label="Guidance Scale [1 - 20]", minimum=1.0, maximum=20.0, value=DIG_PICTURE_GUIDANCE_SCALE)
            steps_input = gr.Number(label="Inference Steps [20 - 100]", minimum=20, maximum=100, value=DIG_PICTURE_NUM_INFERENCE_STEPS)
            width_input = gr.Number(label="Width", minimum=64, maximum=2048, value=DIG_PICTURE_WIDTH)
            height_input = gr.Number(label="Height", minimum=64, maximum=2048, value=DIG_PICTURE_HEIGHT)
        with gr.Row():
            save_location_input = gr.Textbox(label="Image Save Subfolder", value=DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION)
            filename_prefix_input = gr.Textbox(label="Filename Prefix", value=DIG_WEBUI_FILENAME)

    with gr.Accordion("Configuration Save", open=False):
        with gr.Row():
            thread_selector = gr.Dropdown(label="Select Thread")
            with gr.Column(scale=0.5):
                save_thread_btn = gr.Button("💾 Save as NEW Thread")
                update_thread_btn = gr.Button("🔁 Update Current Thread")
            delete_thread_btn = gr.Button("🗑️ Delete Current Thread")

    thread_name_input = gr.Textbox(label="New Thread Name", placeholder="Enter new thread name", visible=False)
    
    generate_btn = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image")
    output_text = gr.Textbox(label="Status", interactive=False)

    inputs_list = [
        prompt_input, neg_prompt_input, guidance_input, steps_input,
        width_input, height_input, filename_prefix_input, save_location_input
    ]

    # Interactions
    generate_btn.click(fn=generate_image, inputs=inputs_list, outputs=[output_image, output_text])
    prompt_input.submit(fn=generate_image, inputs=inputs_list, outputs=[output_image, output_text])

    save_thread_btn.click(lambda: gr.update(visible=True), None, thread_name_input)
    thread_name_input.submit(
        fn=handle_save_thread,
        inputs=[thread_name_input] + inputs_list,
        outputs=[thread_selector, output_text, thread_name_input]
    )

    update_thread_btn.click(
        fn=handle_update_thread,
        inputs=[thread_selector] + inputs_list,
        outputs=output_text
    )

    delete_thread_btn.click(
        fn=handle_delete_thread,
        inputs=thread_selector,
        outputs=[thread_selector, output_text]
    )

    thread_selector.change(
        fn=handle_load_thread,
        inputs=thread_selector,
        outputs=inputs_list
    )

    # Automatically load threads when app launches
    image_gen.load(fn=load_threads, inputs=None, outputs=thread_selector)


image_gen.launch()
