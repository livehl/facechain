from PIL import Image
import torch
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline


def up_4x_pic(prompt: str, img: Image):
    upscaler = StableDiffusionUpscalePipeline.from_pretrained(
        "system_lora/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
    upscaler.to("cuda")
    return upscaler(prompt=prompt, image=img).images[0]


def up_2x_pic(prompt: str, img: Image):
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "system_lora/sd-x2-latent-upscaler", torch_dtype=torch.float16)
    upscaler.to("cuda")
    return upscaler(prompt=prompt, image=img).images[0]
