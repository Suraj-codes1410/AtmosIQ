-- Enable pgvector extension on default database
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create MLflow tracking database if it doesn't exist
SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

-- Connect to mlflow_db and enable vector extension as well
\c mlflow_db;
CREATE EXTENSION IF NOT EXISTS vector;

-- Connect back to primary database
\c atmosiq_db;
