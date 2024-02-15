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


import json
import requests

from celery import Celery

from upscaler.libraries.helper import ImageData


BROKER_URL = 'redis://localhost:9999/0'

celery_app = Celery('tasks', broker=BROKER_URL, backend=BROKER_URL)


@celery_app.task
def predict(image_data: ImageData) -> dict:
    """
    Sends a post request with image_data to prediction endpoint
    then returns the result.
    """
    url = 'http://localhost:10000/predict'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url,
                             headers=headers,
                             json=json.loads(image_data))
    return response.json()