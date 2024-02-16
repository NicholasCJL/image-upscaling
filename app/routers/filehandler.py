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

import asyncio
import io
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import numpy as np

from libraries.helper import (detect_and_convert_image,
                              convert_to_png,
                              convert_image_to_b64,
                              convert_b64_to_image,
                              zip_images)
from libraries.helper import ImageData
from taskqueue.taskqueue import predict


router = APIRouter()

templates = Jinja2Templates(directory="templates", autoescape=False)


@router.get("/download", response_description='zip')
async def download(uuids: list[str] = Query(...)):
    print(uuids)
    images = []
    for uuid in uuids:
        result = predict.AsyncResult(uuid)
        if result.ready():
            result_data = result.get()
            image = convert_b64_to_image(result_data['image'])
            buffer = convert_to_png(np.asarray(image))
            io_buf = io.BytesIO(buffer)
            io_buf.seek(0)
            images.append((io_buf, f"{Path(result_data['name']).stem}.png"))

    headers = {
        'Content-Disposition': 'attachment; filename="images.zip"'
    }

    return_image = zip_images(images)
    return StreamingResponse(return_image, headers=headers)


@router.get("/status")
async def status(uuid: str):
    result = predict.AsyncResult(uuid)
    return JSONResponse({"status": result.ready(), "uuid": uuid})


@router.post("/upload-files")
async def upload_files(request: Request, files: list[UploadFile]):
    read_tasks = []
    
    for file in files:
        contents = file.read()
        print(file.filename)
        read_tasks.append(contents)

    uploaded_files = await asyncio.gather(*read_tasks)
    
    print(len(uploaded_files))

    for file in uploaded_files:
        print(type(file), len(file))
    
    uploaded_images = [detect_and_convert_image(file) for file in uploaded_files]

    image_data = []
    for file, image in zip(files, uploaded_images):
        width, height = image.size
        max_res = max(width, height)
        # limit to 720p
        scale = 720 / max_res if max_res > 720 else 1
        image = image.resize((int(width * scale), int(height * scale)))
        image_b64 = convert_image_to_b64(image)
        image_data.append(ImageData(image=image_b64, name=file.filename))
        
    result_uuids = [predict.delay(image_datum.model_dump_json()).task_id
                    for image_datum in image_data]

    return JSONResponse({"uuids": result_uuids})
