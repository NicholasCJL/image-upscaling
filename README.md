# Simple Image Upscaling

This is a simple image upscaling app that is designed to run with minimal configuration on docker. By default, the upscaling model uses the Stability AI Stable Diffusion x4 upscaler available on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler).

Due to the nature of the model, using a GPU is **highly recommended**.

## Requirements
This repository is designed to run with docker compose on a machine with an nvidia GPU with compute capability >=3.5 with drivers supporting CUDA >=11.8. If you have an nvidia GPU with the Maxwell architecture (GTX 9xx series) and newer with at least 8GB of VRAM, you should be able to run this.

The CUDA version packaged with the docker containers can be modified if needed by modifying the base image in `upscaler/Dockerfile`. If you would like to experiment running with a CPU, simply remove the GPU binding in `docker-compose.yml`.

## Quick Start

1. Clone this git repository and enter the directory:

```
git clone https://github.com/NicholasCJL/image-upscaling.git
cd image-upscaling
```

2. Spin up the containers:

```
docker compose up
```

3. Access the webapp at `http://localhost:9876`

## Using a different model
To use a different model,

1. modify the model downloading in `upscaler/download_model.py` and
2. the model pipeline in `upscaler/main.py` for both the model loading and pipeline inference.

The relevant code for model loading is in the `lifespan` method, and the relevant code for pipeline inference is in the `upscaled_image = ml_models["upscaler"]` function call.

## Things To Note
The upscaler works best with small images of both dimensions ~360 to 480p. The images do not need to be square.

Larger images will work, but they require huge amounts of VRAM that quickly become intractable. The time taken for the upscaling also increases with image resolution.