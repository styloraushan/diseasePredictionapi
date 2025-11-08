# Disease Suggestion API based on Symptoms 

This API gives disease suggestion based on given set of symptoms using a trained Logistic Regression model. It uses WordNet-based synonym expansion, lemmatization and Vectorization: Symptom presence binary encoding

### Build & Run the Docker Container

1. **Create Volumne:**
   ```sh
   docker volume create symptom_data
   ```
3. **Build Docker Image:**
   ```sh
   docker build -t styloraushan10727/demo-python:0.0.3.RELEASE .
   ```
4. **Run Docker Container: Run the container and mount the volume to persist dataset updates**
     ```sh
     docker run -d -p 5000:5000 -v symptom_data:/app/Dataset styloraushan10727/demo-python:0.0.3.RELEASE
    ```
5. **Verify Dataset Access:To check the dataset inside the running container:**
   ```sh
   docker exec -it <container_id> tail /app/Dataset/diseasesymp_updated.csv
   ```



