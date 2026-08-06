#!/usr/bin/env bash
set -e

echo "=== Starting atmosIQ Infrastructure Services (PostgreSQL, pgAdmin, MLflow) ==="

if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

docker compose up -d

echo "Waiting for PostgreSQL and MLflow services to be ready..."
docker compose ps

echo "=== Infrastructure Started Successfully ==="
echo "PostgreSQL: localhost:5432 (User: postgres, DB: atmosiq_db)"
echo "pgAdmin:    http://localhost:5050 (Email: admin@atmosiq.com)"
echo "MLflow:     http://localhost:5000"
