"""
    Copyright (C) 2024 Nicholas Chong (contact@nicholascjl.dev)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import sys

sys.dont_write_bytecode = True

from contextlib import asynccontextmanager
import gc

from diffusers import StableDiffusionUpscalePipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch

from upscaler.libraries.helper import (convert_image_to_b64,
                                       convert_b64_to_image,
                                       ImageData)


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id,
                                                              torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    pipeline.set_use_memory_efficient_attention_xformers(True)

    # Load the ML model
    ml_models["upscaler"] = pipeline

    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        torch.cuda.empty_cache()


@app.post("/predict")
async def predict(image_data: ImageData):
    """
    Predicts the output of the image.
    """
    image = convert_b64_to_image(image_data.image)
    with ClearCache():
        upscaled_image = ml_models["upscaler"](prompt="",
                                               image=image,
                                               num_inference_steps=20).images[0]
    return JSONResponse({'image': convert_image_to_b64(upscaled_image),
                         'name': image_data.name})
