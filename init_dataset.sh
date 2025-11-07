#!/bin/bash
set -e

echo "Starting container initialization..."

# If mounted /app/Dataset is empty, copy from backup

if [ ! -f "/app/Dataset/diseasesymp_updated.csv" ]; then
  echo " Initializing dataset volume from backup..."

  mkdir -p /app/Dataset
  cp -r /app/Dataset_backup/* /app/Dataset/ 2>/dev/null || true
else
  echo "Existing dataset detected — skipping initialization."
fi

echo "Launching Flask API..."
python app.py
