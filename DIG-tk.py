# Written by StormTheory

import customtkinter as ctk
from tkinter import messagebox, simpledialog
from PIL import ImageTk, Image
from datetime import datetime
import torch
import os
import json
import re
import gc
import threading

# --- Globals ---
uploaded_image = None
uploaded_path = None
last_output_path = None  # 🧠 Track last saved image for deletion

MAX_TOKEN = 77
CHARACTER_LIMIT = MAX_TOKEN * 3

ICON_PATH = "assets/DIG_icon.png"

# ⚙️ Memory management environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# Run diffusers and transformers in offline mode
from transformers.utils import logging
logging.set_verbosity_error()
os.environ["TRANSFORMERS_OFFLINE"] = "1"


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

def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def delete_last_output():
    global last_output_path
    if last_output_path and os.path.isfile(last_output_path):
        try:
            os.remove(last_output_path)
            status_var.set(f"🗑 Deleted: {os.path.basename(last_output_path)}")
            last_output_path = None  # 🧠 Clear saved reference
            image_label.configure(image="", text="")  # Clear preview
        except Exception as e:
            status_var.set(f"❌ Delete error: {e}")
    else:
        status_var.set("⚠️ No saved image to delete.")

def delete_regen():
    global last_output_path
    if last_output_path and os.path.isfile(last_output_path):
        try:
            os.remove(last_output_path)
            status_var.set(f"🗑 Deleted: {os.path.basename(last_output_path)}")
            last_output_path = None  # 🧠 Clear saved reference
            image_label.configure(image="", text="")  # Clear preview
        except Exception as e:
            status_var.set(f"❌ Delete error: {e}")
    else:
        status_var.set("⚠️ No saved image to delete.")
    generate_image()

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
        status_label.configure(text_color=colors[index % len(colors)])
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
    generate_button.configure(state="disabled")
    start_flashing()
    threading.Thread(target=threaded_generate, daemon=True).start()

def threaded_generate():
    global last_output_path
    last_output_path = ""
    print("🧵 Preset started")
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
        
        # 🧼 Clean image (strip EXIF + auto-orient and Meta Data for safety & display consistency)
        from PIL import ImageOps
        clean_image = ImageOps.exif_transpose(result).convert("RGB")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.png")
        clean_image.save(filepath)
        last_output_path = filepath  # 🧠 Save reference to last output path

        def update_ui():
            status_var.set(f"✅ Saved: {filepath}")
            #imgtk = ImageTk.PhotoImage(result.convert("RGB").resize((256, 256)))
            #image_label.configure(image=imgtk)
            #image_label.image = imgtk
            
            ### After Warning on terminal
            preview_img = Image.open(filepath).convert("RGB").resize((256, 256))  # or use result directly if already in memory
            ctk_img = ctk.CTkImage(light_image=preview_img, size=(256, 256))
            image_label.configure(image=ctk_img)
            image_label.image = ctk_img

            stop_flashing()
            generate_button.configure(state="normal")

        root.after(0, update_ui)

    except Exception as e:
        print(f"❌ Exception: {e}")
        root.after(0, lambda e=e: (
            status_var.set(f"❌ Error: {str(e)}"),
            stop_flashing(),
            generate_button.configure(state="normal")
        ))


def create_new_preset():
    # Modal dialog for new preset name input
    new_thread_name = simpledialog.askstring("New Preset", "Enter preset name:", parent=root)
    if not new_thread_name:
        return
    new_thread_name = new_thread_name.strip()

    # Check if preset already exists
    if new_thread_name in list_threads():
        messagebox.showinfo("Preset Exists", "⚠️ A preset with that name already exists.", parent=root)
        return

    # Save the current config as the new preset to initialize it
    current_config = {
        "prompt": prompt_var.get(),
        "neg_prompt": neg_prompt_var.get(),
        "guidance_scale": float(guidance_scale_var.get()),
        "steps": int(steps_var.get()),
        "width": int(width_var.get()),
        "height": int(height_var.get()),
        "filename_prefix": filename_prefix_var.get(),
        "save_location": save_location_var.get()
    }
    save_config(new_thread_name, current_config)

    # Update combo box options and set the new preset as active
    thread_menu.configure(values=list_threads())
    thread_var.set(new_thread_name)

    # Load the new preset to update UI fields accordingly
    load_thread()

    status_var.set(f"✅ New preset '{new_thread_name}' created and selected.")


# 💾 Save and load thread configurations
def update_thread():
    name = thread_menu.get().strip()
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
    #thread_menu.configure(values=list_threads())
    #thread_menu.set(name)
    status_var.set(f"✅ Preset '{name}' saved.")

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

######################################################################################

# Max character limit enforcement for Textbox (manual method)
def limit_input_length(new_text: str) -> bool:
    return len(new_text) <= CHARACTER_LIMIT  # Limit to input characters

######################################################################################
def delete_thread_ui():
    name = thread_var.get()

    # 🚫 Prevent deletion of default thread
    if name == DEFAULT_THREAD_NAME:
        status_var.set("⚠️ Cannot delete the default thread.")
        return

    # 🪟 Confirmation popup
    popup = ctk.CTkToplevel(root)
    popup.title("Confirm Delete")
    popup.geometry("320x160")

    # 🧱 Layout components inside popup
    message = ctk.CTkLabel(
        popup,
        text=f"Are you sure you want to delete\nthread '{name}'?",
        font=ctk.CTkFont(size=14),
        justify="center"
    )
    message.pack(pady=20)

    # 👉 Action buttons (Cancel / Confirm)
    button_frame = ctk.CTkFrame(popup)
    button_frame.pack(pady=10)

    def cancel_deletion():
        popup.destroy()

    def confirm_deletion():
        if delete_thread(name):
            thread_menu.configure(values=list_threads())
            thread_menu.set(DEFAULT_THREAD_NAME)
            status_var.set(f"🗑️ Deleted '{name}'")
            load_thread()
        else:
            status_var.set("⚠️ Deletion failed.")
        popup.destroy()

    ctk.CTkButton(button_frame, text="❌ Cancel", command=cancel_deletion, width=100).pack(side="left", padx=10)
    ctk.CTkButton(button_frame, text="🗑️ Confirm", command=confirm_deletion, fg_color="red", hover_color="#aa0000", width=100).pack(side="left", padx=10)

    # ✅ Delay grab until window is rendered
    popup.after(100, popup.grab_set)

################################################################################################


# 🖼️ GUI setup using CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

root = ctk.CTk()
root.title(DIG_WEBUI_TITLE)
root.geometry("700x870")

# 🧠 Variables
prompt_var = ctk.StringVar()
neg_prompt_var = ctk.StringVar()
guidance_scale_var = ctk.StringVar(value=str(DIG_PICTURE_GUIDANCE_SCALE))
steps_var = ctk.StringVar(value=str(DIG_PICTURE_NUM_INFERENCE_STEPS))
width_var = ctk.StringVar(value=str(DIG_PICTURE_WIDTH))
height_var = ctk.StringVar(value=str(DIG_PICTURE_HEIGHT))
filename_prefix_var = ctk.StringVar(value=DIG_WEBUI_FILENAME)
save_location_var = ctk.StringVar(value=DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION)
thread_name_var = ctk.StringVar()
thread_var = ctk.StringVar()
status_var = ctk.StringVar()

# 🧠 UI Layout  ########################################


ctk.CTkLabel(root, text=DIG_WEBUI_TOP_PAGE_BANNER, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

prompt_frame = ctk.CTkFrame(root)
prompt_frame.pack(padx=10, pady=10)

thread_frame = ctk.CTkFrame(prompt_frame)
thread_frame.pack(pady=3, anchor="center")

ctk.CTkLabel(thread_frame, text="Preset", width=150, anchor="e").pack(side="left")
thread_menu = ctk.CTkComboBox(thread_frame, variable=thread_var, values=list_threads(), command=load_thread)
thread_menu.pack(side="left", padx=(10, 0))


# 🔤 Prompt input row with label and entry
prompt_row = ctk.CTkFrame(prompt_frame)
prompt_row.pack(pady=0, anchor="center")

ctk.CTkLabel(prompt_row, text="Prompt", width=150, anchor="e").pack(side="left")

# Register the validation command with the parent frame
vcmd = prompt_frame.register(limit_input_length)

# 🧠 Prompt Entry with input length validation
prompt_entry = ctk.CTkEntry(
    prompt_row,
    textvariable=prompt_var,
    width=500,
    validate="key",
    validatecommand=(vcmd, '%P')
)
prompt_entry.pack(side="left", padx=(10, 0))
prompt_entry.bind("<Return>", lambda event: generate_image())


status_label = ctk.CTkLabel(prompt_frame, textvariable=status_var, text_color="green")
status_label.pack()


#############################################################################################

neg_frame = ctk.CTkFrame(root)
neg_frame.pack(padx=10, pady=10, fill="x")

# 🔤 Negative Prompt input row with left-aligned label
neg_prompt_row = ctk.CTkFrame(neg_frame)
neg_prompt_row.pack(pady=4, anchor="center")

ctk.CTkLabel(neg_prompt_row, text="Negative Prompt", width=150, anchor="e").pack(side="left")

ctk.CTkEntry(neg_prompt_row, textvariable=neg_prompt_var, width=500).pack(side="left", padx=(10, 0))

###############################################################################
advanced_frame = ctk.CTkFrame(root)
advanced_frame.pack(padx=10, pady=10, fill="x")

# 🔧 Advanced Inputs
# Guidance Scale row
guidance_frame = ctk.CTkFrame(advanced_frame)  # Row container
guidance_frame.pack(pady=3, anchor="center")  # Center anchored

ctk.CTkLabel(guidance_frame, text="Guidance Scale [1 - 20]", width=150, anchor="e").pack(side="left")
ctk.CTkEntry(guidance_frame, textvariable=guidance_scale_var, width=180).pack(side="left", padx=(10, 0))

# Steps row
steps_frame = ctk.CTkFrame(advanced_frame)
steps_frame.pack(pady=3, anchor="center")

ctk.CTkLabel(steps_frame, text="Steps [20 - 100]", width=150, anchor="e").pack(side="left")
ctk.CTkEntry(steps_frame, textvariable=steps_var, width=180).pack(side="left", padx=(10, 0))

############################################################################################

# Width row
width_frame = ctk.CTkFrame(advanced_frame)
width_frame.pack(pady=3, anchor="center")

ctk.CTkLabel(width_frame, text="Width", width=150, anchor="e").pack(side="left")
ctk.CTkEntry(width_frame, textvariable=width_var, width=180).pack(side="left", padx=(10, 0))

# Height row
height_frame = ctk.CTkFrame(advanced_frame)
height_frame.pack(pady=3, anchor="center")

ctk.CTkLabel(height_frame, text="Height", width=150, anchor="e").pack(side="left")
ctk.CTkEntry(height_frame, textvariable=height_var, width=180).pack(side="left", padx=(10, 0))

#########################################

# Save Subfolder row
save_frame = ctk.CTkFrame(advanced_frame)
save_frame.pack(pady=3, anchor="center")

ctk.CTkLabel(save_frame, text="Save Subfolder", width=150, anchor="e").pack(side="left")
ctk.CTkEntry(save_frame, textvariable=save_location_var, width=180).pack(side="left", padx=(10, 0))

filename_frame = ctk.CTkFrame(advanced_frame)
filename_frame.pack(pady=3, anchor="center")

ctk.CTkLabel(filename_frame, text="Filename Prefix", width=150, anchor="e").pack(side="left")
ctk.CTkEntry(filename_frame, textvariable=filename_prefix_var, width=180).pack(side="left", padx=(10, 0))



#ctk.CTkLabel(root, text="Preset").pack()
#thread_menu = ctk.CTkComboBox(root, variable=thread_var, values=list_threads(), command=load_thread)
#thread_menu.pack(pady=4)

# 🧱 Row frame for the three preset buttons
preset_button_frame = ctk.CTkFrame(root)
preset_button_frame.pack(pady=10)

# ➕ New Preset button
btn_new = ctk.CTkButton(preset_button_frame, text="➕ New Preset", command=create_new_preset)
btn_new.pack(side="left", padx=5)

# 💾 Update Preset button
btn_update = ctk.CTkButton(preset_button_frame, text="💾 Update Preset", command=update_thread)
btn_update.pack(side="left", padx=5)

# 🗑️ Delete Preset button
btn_delete = ctk.CTkButton(preset_button_frame, text="🗑️ Delete Preset", command=delete_thread_ui)
btn_delete.pack(side="left", padx=5)

generate_button = ctk.CTkButton(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=5)

image_label = ctk.CTkLabel(root, text="")
image_label.pack()

# 🗑 Add delete button below strength
delete_btn = ctk.CTkButton(
    root,
    text="🗑 Delete Last Output",
    command=delete_last_output,
    fg_color="#b22222",  # firebrick red
    hover_color="#8b0000"
)
delete_btn.pack(pady=(10, 0))

# 🗑 Add delete and Regen button below strength
delete_regen_btn = ctk.CTkButton(
    root,
    text="🗑 Delete & 🔄 Regenerate",
    command=delete_regen,
    fg_color="#b22222",  # firebrick red
    hover_color="#8b0000"
)
delete_regen_btn.pack(pady=(10, 0))

# 🖼️ Icon
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
