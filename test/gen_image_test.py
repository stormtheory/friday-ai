#from transformers import CLIPImageProcessor
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

prompt = "a samurai standing on Mars with a red sun"
image = pipe(prompt).images[0]
image.save("test.png")
print("✅ Image saved as test.png")
