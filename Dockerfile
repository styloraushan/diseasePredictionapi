FROM python:3.12.6-slim
WORKDIR /app
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY Dataset/ ./Dataset_backup/
COPY init_dataset.sh ./
RUN chmod +x init_dataset.sh
EXPOSE 5000
ENTRYPOINT ["bash", "init_dataset.sh"]
