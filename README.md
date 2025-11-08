

### Create Docker Volume  
This volume ensures that dataset changes persist even after the container is stopped or removed.  



```bash
docker volume create symptom_data

### Build Docker Image
Build the image from your Dockerfile:  

```bash
docker build -t styloraushan10727/demo-python:0.0.3.RELEASE .

### Run Docker Container 
Run the container and mount the volume to persist dataset updates:  

```bash
docker run -d -p 5000:5000 -v symptom_data:/app/Dataset styloraushan10727/demo-python:0.0.3.RELEASE

### Verify Dataset Access
To check the dataset inside the running container: 

```bash
docker exec -it <container_id> tail /app/Dataset/diseasesymp_updated.csv


