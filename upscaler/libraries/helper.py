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
from PIL import Image
from pydantic import BaseModel


class ImageData(BaseModel):
    """
    Contains image data in base64 format with filename.
    """
    image: str
    name: str


def convert_image_to_b64(image: Image) -> str:
    # Converts Image to base64 string
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def convert_b64_to_image(b64_string: str) -> Image:
    # Converts base64 string to image
    print(type(b64_string))
    print(len(b64_string))
    print(b64_string[:100])
    return Image.open(BytesIO(base64.b64decode(b64_string)))