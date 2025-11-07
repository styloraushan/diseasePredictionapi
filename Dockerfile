# # Use lightweight Python image
# FROM python:3.12-slim

# # Set working directory
# WORKDIR /app

# # Copy dependency list and install
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the app code
# COPY . .

# # Create Dataset directory (in case volume is empty)
# RUN mkdir -p /app/Dataset

# # Ensure Flask can write to Dataset directory
# RUN chmod -R 777 /app/Dataset

# # Environment variables for Flask
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_ENV=production

# # Expose Flask port
# EXPOSE 6000

# # Run the Flask app
# CMD ["python", "app.py"]

# Use official Python image



# FROM python:3.12.6-slim

# # Set working directory
# WORKDIR /app

# # Copy dependency file and install
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code and assets into image
# COPY app.py .
# COPY Dataset/diseasesymp_updated.csv ./Dataset/
# COPY model.pkl .

# # Expose port 5000
# EXPOSE 5000

# # Command to run Flask app
# CMD ["python", "app.py"]

# ---------- Base Image ----------
FROM python:3.12.6-slim

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Install Dependencies ----------
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

# ---------- Copy Requirements & Install ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Application Code ----------
COPY app.py ./

# Make sure both the source and destination are directories
COPY Dataset/ ./Dataset_backup/

# ---------- Copy Initialization Script ----------
COPY init_dataset.sh ./
RUN chmod +x init_dataset.sh

# ---------- Expose Flask Port ----------
EXPOSE 5000

# ---------- Entry Point ----------
ENTRYPOINT ["bash", "init_dataset.sh"]
