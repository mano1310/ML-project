FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
RUN pip install --no-cache-dir .
RUN python -m churnlens.train
EXPOSE 8000
CMD ["uvicorn", "churnlens.api:app", "--host", "0.0.0.0", "--port", "8000"]