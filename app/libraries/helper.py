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

import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
import zipfile

import cv2
import numpy as np
from PIL import Image


class ImageData(BaseModel):
    """
    Contains image data in base64 format with filename.
    """
    image: str
    name: str


def detect_and_convert_image(file):
    nparr = np.fromstring(file, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def convert_to_png(image: np.ndarray) -> bytes:
    # Converts numpy array of image to png format
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    return buffer

def convert_image_to_b64(image: Image) -> str:
    # Converts Image to base64 string
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def convert_b64_to_image(b64_string: str) -> Image:
    # Converts base64 string to image
    return Image.open(BytesIO(base64.b64decode(b64_string)))

def zip_images(images: list[tuple[BytesIO, str]]):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        for _, (image, name) in enumerate(images):
            zip_file.writestr(f'{Path(name).stem}.png', image.getvalue())

    zip_buffer.seek(0)
    return zip_buffer
