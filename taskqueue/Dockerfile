FROM python:3.10-alpine3.18

WORKDIR /app

COPY . .

RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

CMD ["celery", "-A", "taskqueue", "worker", "--pool=solo", "--loglevel=info"]