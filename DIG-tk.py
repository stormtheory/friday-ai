# Written by StormTheory
# Fixed and Upgraded by ChatGPT (OpenAI)

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
from datetime import datetime
import torch
import getpass
import os
import json
import re
import gc
import threading
from accelerate import PartialState

# ⚙️ Memory management environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print(f"🧠 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔋 VRAM available: {torch.cuda.mem_get_info()[0] / 1024**2:.2f} MB")


# 🧠 Model and config import
from diffusers import StableDiffusionXLPipeline
from config import (
    DIG_DEFAULT_PROMPT as DEFAULT_PROMPT,
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

ICON_PATH = "assets/DIG_icon.png"

def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

DEFAULT_THREAD_NAME = "Default"
os.makedirs(DIG_WEBUI_THREAD_DATA_DIR, exist_ok=True)

# ✅ Load model directly to GPU for stability
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

# 🧠 Ensure default config exists
default_config_path = os.path.join(DIG_WEBUI_THREAD_DATA_DIR, DEFAULT_THREAD_NAME + ".json")
if not os.path.exists(default_config_path):
    with open(default_config_path, "w") as f:
        json.dump(DEFAULT_CONFIG, f)

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
    files = os.listdir(DIG_WEBUI_THREAD_DATA_DIR)
    return sorted(set([DEFAULT_THREAD_NAME] + [f[:-5] for f in files if f.endswith(".json")]))

def delete_thread(name):
    if name == DEFAULT_THREAD_NAME:
        return False
    path = os.path.join(DIG_WEBUI_THREAD_DATA_DIR, sanitize_filename(name) + ".json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

# 🔁 Flashing UI while generating
flash_job = None

def start_flashing():
    global flash_job
    colors = ["green", "orange"]
    index = 0

    def flash():
        nonlocal index
        status_var.set("🚧 Generating...")
        status_label.config(fg=colors[index % len(colors)])
        index += 1
        globals()["flash_job"] = root.after(500, flash)

    flash()

def stop_flashing():
    global flash_job
    if flash_job:
        root.after_cancel(flash_job)
        flash_job = None

# 🔄 Generation thread logic
def generate_image():
    generate_button.config(state=tk.DISABLED)
    start_flashing()
    threading.Thread(target=threaded_generate, daemon=True).start()

def threaded_generate():
    print("🧵 Thread started")
    try:
        torch.cuda.empty_cache()

        prompt = prompt_var.get().strip() or DEFAULT_PROMPT
        neg_prompt = neg_prompt_var.get().strip() or DIG_PICTURE_NEG_PROMPT
        gs = float(guidance_scale_var.get() or DIG_PICTURE_GUIDANCE_SCALE)
        steps = int(steps_var.get() or DIG_PICTURE_NUM_INFERENCE_STEPS)
        width = int(width_var.get() or DIG_PICTURE_WIDTH)
        height = int(height_var.get() or DIG_PICTURE_HEIGHT)
        filename_prefix = filename_prefix_var.get().strip() or DIG_WEBUI_FILENAME
        save_loc = save_location_var.get().strip() or DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION

        # 📏 Validate image dimensions
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("❌ Width and Height must be divisible by 8.")

        save_dir = os.path.expanduser(f"~/{save_loc}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"🧠 Generating: {prompt}")
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=gs,
            num_inference_steps=steps,
            width=width,
            height=height
        ).images[0]
        print("✅ Generation complete")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.png")
        result.save(filepath)

        def update_ui():
            status_var.set(f"✅ Saved: {filepath}")
            imgtk = ImageTk.PhotoImage(result.convert("RGB").resize((256, 256)))
            image_label.configure(image=imgtk)
            image_label.image = imgtk
            stop_flashing()
            generate_button.config(state=tk.NORMAL)

        root.after(0, update_ui)

    except Exception as e:
        print(f"❌ Exception: {e}")
        root.after(0, lambda e=e: (
            status_var.set(f"❌ Error: {str(e)}"),
            stop_flashing(),
            generate_button.config(state=tk.NORMAL)
        ))

# 💾 Save and load thread configurations
def save_thread():
    name = thread_name_var.get().strip()
    if not name or name == DEFAULT_THREAD_NAME:
        status_var.set("⚠️ Invalid thread name.")
        return

    config = {
        "prompt": prompt_var.get(),
        "neg_prompt": neg_prompt_var.get(),
        "guidance_scale": float(guidance_scale_var.get()),
        "steps": int(steps_var.get()),
        "width": int(width_var.get()),
        "height": int(height_var.get()),
        "filename_prefix": filename_prefix_var.get(),
        "save_location": save_location_var.get()
    }
    save_config(name, config)
    thread_menu["values"] = list_threads()
    thread_menu.set(name)
    status_var.set(f"✅ Thread '{name}' saved.")

def load_thread(event=None):
    name = thread_var.get()
    cfg = load_config(name)
    prompt_var.set(cfg["prompt"])
    neg_prompt_var.set(cfg["neg_prompt"])
    guidance_scale_var.set(cfg["guidance_scale"])
    steps_var.set(cfg["steps"])
    width_var.set(cfg["width"])
    height_var.set(cfg["height"])
    filename_prefix_var.set(cfg["filename_prefix"])
    save_location_var.set(cfg["save_location"])

def delete_thread_ui():
    name = thread_var.get()
    if delete_thread(name):
        thread_menu["values"] = list_threads()
        thread_menu.set(DEFAULT_THREAD_NAME)
        status_var.set(f"🗑️ Deleted '{name}'")
        load_thread()
    else:
        status_var.set("⚠️ Cannot delete this thread.")

# 🖼️ GUI setup
root = tk.Tk()
root.title(DIG_WEBUI_TITLE)
root.geometry("650x1100")

prompt_var = tk.StringVar()
neg_prompt_var = tk.StringVar()
guidance_scale_var = tk.StringVar(value=str(DIG_PICTURE_GUIDANCE_SCALE))
steps_var = tk.StringVar(value=str(DIG_PICTURE_NUM_INFERENCE_STEPS))
width_var = tk.StringVar(value=str(DIG_PICTURE_WIDTH))
height_var = tk.StringVar(value=str(DIG_PICTURE_HEIGHT))
filename_prefix_var = tk.StringVar(value=DIG_WEBUI_FILENAME)
save_location_var = tk.StringVar(value=DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION)
thread_name_var = tk.StringVar()
thread_var = tk.StringVar()
status_var = tk.StringVar()

tk.Label(root, text=DIG_WEBUI_TOP_PAGE_BANNER, font=("Arial", 16)).pack(pady=10)
tk.Label(root, text="Prompt").pack()
prompt_entry = tk.Entry(root, textvariable=prompt_var, width=80)
prompt_entry.pack()
prompt_entry.bind("<Return>", lambda event: generate_image())

status_label = tk.Label(root, textvariable=status_var, fg="green")
status_label.pack()
#tk.Label(root, textvariable=status_var, fg="green").pack()

frame = tk.LabelFrame(root, text="Advanced Settings")
frame.pack(fill="x", padx=10, pady=5)

tk.Label(frame, text="Negative Prompt").pack()
tk.Entry(frame, textvariable=neg_prompt_var, width=80).pack()

for label, var in [("Guidance Scale", guidance_scale_var), ("Steps", steps_var),
                   ("Width", width_var), ("Height", height_var),
                   ("Save Subfolder", save_location_var), ("Filename Prefix", filename_prefix_var)]:
    tk.Label(frame, text=label).pack()
    tk.Entry(frame, textvariable=var, width=30).pack()

tk.Label(root, text="Thread").pack()
thread_menu = ttk.Combobox(root, textvariable=thread_var, values=list_threads(), state="readonly")
thread_menu.pack()
thread_menu.bind("<<ComboboxSelected>>", load_thread)

tk.Entry(root, textvariable=thread_name_var, width=30).pack(pady=3)
tk.Button(root, text="💾 Save Thread", command=save_thread).pack()
tk.Button(root, text="🗑️ Delete Thread", command=delete_thread_ui).pack(pady=2)

generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=5)

image_label = tk.Label(root)
image_label.pack()

if os.path.exists(ICON_PATH):
    try:
        icon_img = ImageTk.PhotoImage(Image.open(ICON_PATH))
        root.iconphoto(True, icon_img)
    except Exception as e:
        print(f"⚠️ Failed to set window icon: {e}")

thread_var.set(DEFAULT_THREAD_NAME)
load_thread()
prompt_entry.focus_set()
root.mainloop()
