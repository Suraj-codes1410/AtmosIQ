#!/usr/bin/env bash
set -e

echo "=== Starting atmosIQ Phase 1 Data Engineering Pipeline ==="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ensure log directory exists
mkdir -p ml/data/logs ml/data/raw ml/data/processed

echo "1/5 Running Open-Meteo Ingestion Script..."
python3 -m ml.src.ingestion.open_meteo_ingestion

echo "2/5 Running NASA FIRMS Fire Hotspot Ingestion Script..."
python3 -m ml.src.ingestion.nasa_firms_ingestion

echo "3/5 Running OpenAQ Pollutant Ingestion Script..."
python3 -m ml.src.ingestion.openaq_ingestion

echo "4/5 Running Calendar Feature Generator Script..."
python3 -m ml.src.ingestion.calendar_ingestion

echo "5/5 Running Master Dataset Validation & Merge Pipeline..."
python3 -m ml.src.preprocessing.merge_pipeline

echo "=== Data Engineering Pipeline Complete ==="
echo "Master Dataset Output: ml/data/processed/master_dataset.csv"
echo "Log File:              ml/data/logs/data_engineering.log"
