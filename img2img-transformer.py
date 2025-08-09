# AI Image-to-Image Generator
# By StormTheory – Privacy-focused, GPU-efficient

import os
from pathlib import Path
import datetime
import gc

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk

from config import (
    DIG_DEFAULT_PROMPT,
    DIG_PICTURE_NEG_PROMPT,
    DIG_PICTURE_HEIGHT,
    DIG_PICTURE_NUM_INFERENCE_STEPS,
    DIG_PICTURE_WIDTH,
    DIG_PICTURE_GUIDANCE_SCALE,
    DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION,
    MODELS_DIR
)

# --- Globals ---
uploaded_image = None
uploaded_path = None
last_output_path = None  # 🧠 Track last saved image for deletion
ICON_PATH = "assets/I2I_icon.png"

# Define the model repo ID and cache directory
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
cache_dir = os.path.expanduser(f"{MODELS_DIR}/diffusers/stable-diffusion-xl-base-1.0")

# 🧠 Set PyTorch env early for CUDA memory fragmentation control
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# 🧠 Clean GPU memory before loading
gc.collect()
torch.cuda.empty_cache()

# 🎛️ Init Tkinter and CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

root = ctk.CTk()
root.title("AI Image-to-Image (I2I) Generator")
root.geometry("900x800")

# 🧠 Load stable diffusion XL img2img pipeline
from diffusers import StableDiffusionXLImg2ImgPipeline

if os.path.isdir(cache_dir):
    # ✅ Load model directly to GPU for stability
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        device_map="balanced"
    )
    pipe.reset_device_map()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    print("✅ Loaded model offline from cache.")
else:
    # 🌐 Model not found locally → download & cache automatically
    print("📥 Model not found in cache — downloading...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,  # Will be saved here automatically
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        device_map="balanced"
    )
    # 🛡️ Apply same optimizations after download
    pipe.reset_device_map()
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    print("✅ Model downloaded and cached for future offline use.")



if torch.cuda.is_available():
    print(f"🧠 Using GPU: {torch.cuda.get_device_name(0)}")

# --- Globals ---
uploaded_image = None
uploaded_path = None

image_metadata_var = ctk.StringVar(value="No image loaded")
status_bar_var = ctk.StringVar(value="Status: Idle")
prompt_var = ctk.StringVar(value=DIG_DEFAULT_PROMPT)
neg_prompt_var = ctk.StringVar(value=DIG_PICTURE_NEG_PROMPT)
width_var = ctk.StringVar(value=str(DIG_PICTURE_WIDTH))
height_var = ctk.StringVar(value=str(DIG_PICTURE_HEIGHT))
guidance_scale_var = ctk.StringVar(value=str(DIG_PICTURE_GUIDANCE_SCALE))
steps_var = ctk.StringVar(value=str(DIG_PICTURE_NUM_INFERENCE_STEPS))
strength_var = ctk.StringVar(value="0.7")  # custom default

# --- Upload Image ---
def upload_image():
    global uploaded_image, uploaded_path
    pictures_dir = str(Path.home() / "Pictures")
    path = filedialog.askopenfilename(
        initialdir=pictures_dir,
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")]
    )
    if not path:
        return
    try:
        uploaded_image = Image.open(path).convert("RGB")
        uploaded_path = path
        preview = uploaded_image.resize((256, 256))
        img_preview = ctk.CTkImage(light_image=preview, size=(256, 256))
        input_img_label.configure(image=img_preview)
        input_img_label.image = img_preview
        image_metadata_var.set(f"{uploaded_image.width}x{uploaded_image.height} | {uploaded_image.format}")
    except Exception as e:
        status_bar_var.set(f"⚠️ Error loading image: {e}")

def delete_last_output():
    global last_output_path
    if last_output_path and os.path.isfile(last_output_path):
        try:
            os.remove(last_output_path)
            status_bar_var.set(f"🗑 Deleted: {os.path.basename(last_output_path)}")
            last_output_path = None  # 🧠 Clear saved reference
            output_img_label.configure(image="", text="")  # Clear preview
        except Exception as e:
            status_bar_var.set(f"❌ Delete error: {e}")
    else:
        status_bar_var.set("⚠️ No saved image to delete.")
        
def delete_regen():
    global last_output_path
    if last_output_path and os.path.isfile(last_output_path):
        try:
            os.remove(last_output_path)
            status_bar_var.set(f"🗑 Deleted: {os.path.basename(last_output_path)}")
            last_output_path = None  # 🧠 Clear saved reference
            output_img_label.configure(image="", text="")  # Clear preview
        except Exception as e:
            status_bar_var.set(f"❌ Delete error: {e}")
    else:
        status_bar_var.set("⚠️ No saved image to delete.")
    run_generation()


# --- Generate Image ---
def run_generation():
    global last_output_path
    last_output_path = ""
    status_bar_var.set(f"🤔 Thinking...")
    root.update_idletasks()
    
    global uploaded_image
    if uploaded_image is None:
        status_bar_var.set("⚠️ No image to transform.")
        return
    try:
        torch.cuda.empty_cache()

        # Use defaults if user clears inputs
        prompt = prompt_var.get().strip() or DIG_DEFAULT_PROMPT
        neg_prompt = neg_prompt_var.get().strip() or DIG_PICTURE_NEG_PROMPT
        gs = float(guidance_scale_var.get() or DIG_PICTURE_GUIDANCE_SCALE)
        steps = int(steps_var.get() or DIG_PICTURE_NUM_INFERENCE_STEPS)
        width = int(width_var.get() or DIG_PICTURE_WIDTH)
        height = int(height_var.get() or DIG_PICTURE_HEIGHT)
        strength = float(strength_var.get() or 0.7)
            
        # Resize uploaded image to requested dimensions
        init_image = uploaded_image.resize((width, height))
        
        # Run img2img generation
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=gs,
            num_inference_steps=steps
        ).images[0]

        # Clean image metadata and convert to RGB
        result_rgb = ImageOps.exif_transpose(result).convert("RGB")
        
        # Show preview scaled down
        preview = result_rgb.resize((256, 256))
        out_img = ctk.CTkImage(light_image=preview, size=(256, 256))
        output_img_label.configure(image=out_img)
        output_img_label.image = out_img

        # Save output to user's Pictures directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.expanduser(f"~/{DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION}")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"generated_{timestamp}.png")
        result_rgb.save(out_path)
        last_output_path = out_path  # 🧠 Save reference to last output path

        status_bar_var.set(f"✅ Saved: {os.path.basename(out_path)}")
        root.update_idletasks()

    except Exception as e:
        status_bar_var.set(f"❌ Generation error: {str(e)}")

# --- UI Building ---

main_frame = ctk.CTkFrame(root)
main_frame.pack(padx=10, pady=10, fill="both", expand=True)

display_frame = ctk.CTkFrame(main_frame)
display_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Left panel: Current Image + metadata + upload button
left_panel = ctk.CTkFrame(display_frame)
left_panel.pack(side="left", padx=10)

ctk.CTkLabel(left_panel, text="Current Image", font=ctk.CTkFont(size=16)).pack(pady=(0, 10))
input_img_label = ctk.CTkLabel(left_panel, text="", width=256, height=256)
input_img_label.pack()

ctk.CTkLabel(left_panel, textvariable=image_metadata_var, font=ctk.CTkFont(size=12)).pack(pady=3)
ctk.CTkLabel(left_panel, textvariable=status_bar_var, font=ctk.CTkFont(size=12)).pack(pady=3)


upload_btn = ctk.CTkButton(left_panel, text="📤 Upload Image", command=upload_image)
upload_btn.pack(pady=5)

# Middle "arrow" label
ctk.CTkLabel(display_frame, text="->  ->  ->", font=ctk.CTkFont(size=24)).pack(side="left", padx=20)

# Right panel: Transformed image + options
right_panel = ctk.CTkFrame(display_frame)
right_panel.pack(side="left", padx=10)

ctk.CTkLabel(right_panel, text="Transformed Image", font=ctk.CTkFont(size=16)).pack(pady=(0, 10))
output_img_label = ctk.CTkLabel(right_panel, text="", width=256, height=256)
output_img_label.pack()

options_frame = ctk.CTkFrame(right_panel)
options_frame.pack(pady=10)

def create_option_row(parent, label, variable):
    row = ctk.CTkFrame(parent)
    row.pack(pady=2)
    ctk.CTkLabel(row, text=label, width=120, anchor="e").pack(side="left")
    ctk.CTkEntry(row, textvariable=variable, width=100).pack(side="left")

create_option_row(options_frame, "Width", width_var)
create_option_row(options_frame, "Height", height_var)
create_option_row(options_frame, "Guidance Scale", guidance_scale_var)
create_option_row(options_frame, "Steps", steps_var)
create_option_row(options_frame, "Strength", strength_var)

# 🗑 Add delete button below strength
delete_btn = ctk.CTkButton(
    options_frame,
    text="🗑 Delete Last Output",
    command=delete_last_output,
    fg_color="#b22222",  # firebrick red
    hover_color="#8b0000"
)
delete_btn.pack(pady=(10, 0))

# 🗑 Add delete and Regen button below strength
delete_regen_btn = ctk.CTkButton(
    options_frame,
    text="🗑 Delete & 🔄 Regenerate",
    command=delete_regen,
    fg_color="#b22222",  # firebrick red
    hover_color="#8b0000"
)
delete_regen_btn.pack(pady=(10, 0))

# Prompt section
prompt_section = ctk.CTkFrame(main_frame)
prompt_section.pack(fill="x", pady=10)

prompt_row = ctk.CTkFrame(prompt_section)
prompt_row.pack(pady=2, anchor='w')
ctk.CTkLabel(prompt_row, text="Prompt", width=110, anchor="e").pack(side="left")
prompt_entry = ctk.CTkEntry(prompt_row, textvariable=prompt_var, width=800)
prompt_entry.pack(side="left", padx=5)
# ⌨️ Bind Enter key to trigger image generation
prompt_entry.bind("<Return>", lambda event: run_generation())


neg_row = ctk.CTkFrame(prompt_section)
neg_row.pack(pady=2, anchor='w')
ctk.CTkLabel(neg_row, text="Negative Prompt", width=110, anchor="e").pack(side="left")
neg_entry = ctk.CTkEntry(neg_row, textvariable=neg_prompt_var, width=800)
neg_entry.pack(side="left", padx=5)
neg_entry.bind("<Return>", lambda event: run_generation())

generate_btn = ctk.CTkButton(main_frame, text="🚀 Transform", width=120, command=run_generation)
generate_btn.pack(pady=10)

# 🖼️ Icon
if os.path.exists(ICON_PATH):
    try:
        icon_img = ImageTk.PhotoImage(Image.open(ICON_PATH))
        root.iconphoto(True, icon_img)
    except Exception as e:
        print(f"⚠️ Failed to set window icon: {e}")

root.mainloop()
