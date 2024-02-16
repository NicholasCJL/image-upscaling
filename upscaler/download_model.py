from diffusers import StableDiffusionUpscalePipeline
import torch

if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id,
                                                              torch_dtype=torch.float16,
                                                              use_safetensors=True)
    pipeline.save_pretrained("upscaler")