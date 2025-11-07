# 🩺 Symptom → Disease Prediction API  

This project provides a **machine learning–based API** that predicts possible diseases based on symptoms.  
The API is containerized using **Docker** for portability, consistency, and ease of deployment.  

---

## 🚀 Features  

- Predict diseases from input symptoms  
- Containerized Python Flask app for easy deployment  
- Persistent dataset storage using Docker volumes  
- Lightweight and portable image  

---

## 🧰 Prerequisites  

Before you begin, ensure you have the following installed:  

- [Docker](https://docs.docker.com/get-docker/)  
- Python (optional, for local testing)  

---

## 🏗️ Build & Run the Docker Container  

### 1️⃣ Create Docker Volume  
This volume ensures that dataset changes persist even after the container is stopped or removed.  



```bash
docker volume create symptom_data

### 2️⃣ Build Docker Image
Build the image from your Dockerfile:  

```bash
docker build -t styloraushan10727/demo-python:0.0.3.RELEASE .

### 3️⃣ Run Docker Container 
Run the container and mount the volume to persist dataset updates:  

```bash
docker run -d -p 5000:5000 -v symptom_data:/app/Dataset styloraushan10727/demo-python:0.0.3.RELEASE

### 4️⃣ Verify Dataset Access
To check the dataset inside the running container: 

```bash
docker exec -it <container_id> tail /app/Dataset/diseasesymp_updated.csv


