FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0

RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--port", "9876", "--host", "0.0.0.0"]