FROM python:3.12.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip3 install --root-user-action=ignore --upgrade pip \
    && pip3 install --root-user-action=ignore --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD ["curl", "--fail", "http://localhost:8501/_stcore/health"]

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless", "true"]
