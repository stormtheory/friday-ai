# AI Image-to-Image Generator
# By StormTheory – Privacy-focused, GPU-efficient

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import os
import gc
import datetime
from pathlib import Path
from config import DIG_DEFAULT_PROMPT, DIG_PICTURE_NEG_PROMPT, DIG_PICTURE_HEIGHT, DIG_PICTURE_NUM_INFERENCE_STEPS, DIG_PICTURE_WIDTH, DIG_PICTURE_GUIDANCE_SCALE, DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION


# 🧠 Use the CPU as well
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

# 🧠 Optimize VRAM + load pipeline offline
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# 🎛️ Init Tk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

root = ctk.CTk()
root.title("🧠 AI Image-to-Image Generator")
root.geometry("1200x800")

# 🧠 Model setup
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
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

if torch.cuda.is_available():
    print(f"🧠 Using GPU: {torch.cuda.get_device_name(0)}")

# Global vars
uploaded_image = None
uploaded_path = None
image_metadata_var = ctk.StringVar(value="No image loaded")
prompt_var = ctk.StringVar()
neg_prompt_var = ctk.StringVar()

width_var = ctk.StringVar(value="512")
height_var = ctk.StringVar(value="512")
guidance_scale_var = ctk.StringVar(value="7.5")
steps_var = ctk.StringVar(value="40")
strength_var = ctk.StringVar(value="0.7")

# 📤 Upload
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
        image_metadata_var.set(f"⚠️ Error loading image: {e}")

# 🧠 Generate
def run_generation():
    global uploaded_image
    if uploaded_image is None:
        image_metadata_var.set("⚠️ No image to transform.")
        return
    try:
        torch.cuda.empty_cache()
        prompt = prompt_var.get().strip() or "A futuristic cityscape"
        neg_prompt = neg_prompt_var.get().strip()
        gs = float(guidance_scale_var.get())
        steps = int(steps_var.get())
        width = int(width_var.get())
        height = int(height_var.get())
        strength = float(strength_var.get())

        # Resize uploaded image
        init_image = uploaded_image.resize((width, height))

        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=gs,
            num_inference_steps=steps
        ).images[0]
        
        # 🧼 Clean image (strip EXIF + auto-orient and Meta Data for safety & display consistency)
        from PIL import ImageOps
        result_rgb = ImageOps.exif_transpose(result).convert("RGB")

        preview = result_rgb.resize((256, 256))
        out_img = ctk.CTkImage(light_image=preview, size=(256, 256))
        output_img_label.configure(image=out_img)
        output_img_label.image = out_img
             

        # Save output
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.expanduser(f"~/Pictures/generated_{timestamp}.png")
        result_rgb.save(out_path)
        image_metadata_var.set(f"✅ Saved: {os.path.basename(out_path)}")
    except Exception as e:
        image_metadata_var.set(f"❌ Generation error: {str(e)}")

# UI Building
main_frame = ctk.CTkFrame(root)
main_frame.pack(padx=10, pady=10, fill="both", expand=True)

display_frame = ctk.CTkFrame(main_frame)
display_frame.pack(fill="both", expand=True, padx=10, pady=10)

left_panel = ctk.CTkFrame(display_frame)
left_panel.pack(side="left", padx=10)

ctk.CTkLabel(left_panel, text="Current Image", font=ctk.CTkFont(size=16)).pack(pady=(0, 10))
input_img_label = ctk.CTkLabel(left_panel, width=256, height=256)
input_img_label.pack()

ctk.CTkLabel(left_panel, textvariable=image_metadata_var, font=ctk.CTkFont(size=12)).pack(pady=10)
upload_btn = ctk.CTkButton(left_panel, text="📤 Upload Image", command=upload_image)
upload_btn.pack(pady=5)

ctk.CTkLabel(display_frame, text="->  ->  ->", font=ctk.CTkFont(size=24)).pack(side="left", padx=20)

right_panel = ctk.CTkFrame(display_frame)
right_panel.pack(side="left", padx=10)

ctk.CTkLabel(right_panel, text="Transformed Image", font=ctk.CTkFont(size=16)).pack(pady=(0, 10))
output_img_label = ctk.CTkLabel(right_panel, width=256, height=256)
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

prompt_section = ctk.CTkFrame(main_frame)
prompt_section.pack(fill="x", pady=10)

prompt_row = ctk.CTkFrame(prompt_section)
prompt_row.pack(pady=2)
ctk.CTkLabel(prompt_row, text="Prompt", width=120, anchor="e").pack(side="left")
ctk.CTkEntry(prompt_row, textvariable=prompt_var, width=600).pack(side="left", padx=5)

neg_row = ctk.CTkFrame(prompt_section)
neg_row.pack(pady=2)
ctk.CTkLabel(neg_row, text="Negative Prompt", width=120, anchor="e").pack(side="left")
ctk.CTkEntry(neg_row, textvariable=neg_prompt_var, width=600).pack(side="left", padx=5)

generate_btn = ctk.CTkButton(main_frame, text="🚀 Generate", width=120, command=run_generation)
generate_btn.pack(pady=10)

root.mainloop()
