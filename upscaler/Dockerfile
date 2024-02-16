FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . .

RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# predownload model
RUN python download_model.py

CMD ["uvicorn", "main:app", "--port", "10000", "--host", "0.0.0.0"]