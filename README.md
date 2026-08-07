# atmosIQ: Delhi AQI Source-Signal Attribution & Policy Intelligence Platform

![Phase 0](https://img.shields.io/badge/Phase_0-Infrastructure_Ready-brightgreen)
![Phase 1](https://img.shields.io/badge/Phase_1-Data_Engineering_Complete-brightgreen)
![Phase 2](https://img.shields.io/badge/Phase_2-Feature_Matrix_256_Features-brightgreen)
![Java](https://img.shields.io/badge/Java-21-orange)
![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.3-green)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.15-teal)

## Overview & Platform Goals

**atmosIQ** is a production-grade AI platform built to deliver precise **PM2.5 source-signal attribution** and **policy impact intelligence** for the National Capital Region (NCR) of Delhi.

### What atmosIQ Is NOT:
- atmosIQ is **NOT** a simple AQI forecasting chatbot.

### Core Objectives:
1. **Data Engineering & Ingestion (Phase 1)**: Ingest ambient air quality, satellite active fire hotspots, meteorology, and calendar indicators for Delhi NCR (731 daily records across 2 full years: 2023-2024).
2. **Feature Engineering & Process Modeling (Phase 2)**: Transform raw data into a 256-feature matrix (`feature_dataset.csv`) capturing wind vectors, stubble burning advection scores, chemical pollutant ratios, rolling volatility, and non-linear interactions.
3. **Machine Learning Attribution (Phase 3+)**: Train custom ML regression models (XGBoost, LightGBM) to predict PM2.5 concentrations based on environmental feature groups.
4. **Explainable AI (SHAP)**: Calculate exact Shapley additive feature-group attributions (TreeSHAP) to decompose PM2.5 levels into actionable source signals (vehicular, industrial, stubble burning, meteorology).
5. **Spring AI & Policy RAG Orchestration**: Orchestrate predictions and SHAP attributions with clean air policy documents (GRAP, NCAP) using Spring AI, FastAPI, and PostgreSQL with `pgvector` semantic vector store.

---

## High-Level System Architecture

```mermaid
graph TD
    subgraph Phase1 ["Phase 1: Data Engineering"]
        A1[OpenAQ API / CPCB] --> B1[Raw Pollutant CSV]
        A2[NASA FIRMS Satellite] --> B2[Raw Fire Hotspot CSV]
        A3[Open-Meteo API] --> B3[Raw Weather CSV]
        A4[Calendar Engine] --> B4[Raw Calendar CSV]
        B1 & B2 & B3 & B4 --> C[Data Validation & Master Merge]
        C --> D[processed/master_dataset.csv]
    end

    subgraph Phase2 ["Phase 2: Feature Engineering"]
        D --> E[Time & Calendar Extractor]
        D --> F[Weather & Wind Vector Extractor]
        D --> G[Satellite Fire Hotspot Extractor]
        D --> H[Pollution & Chemical Ratio Extractor]
        E & F & G & H --> I[Interaction & Lags/Rolling Pipeline]
        I --> J[processed/feature_dataset.csv (731 × 256)]
    end

    subgraph Phase3Plus ["Phase 3+: ML Training & Serving (Future)"]
        J --> K[XGBoost / LightGBM Regressors]
        K --> L[SHAP Attribution Engine]
        L --> M[FastAPI Inference Microservice]
        M --> N[Spring Boot 3 + Spring AI Orchestrator]
        N --> O[Frontend Policy Dashboard]
    end
```

---

## Monorepo Folder Structure

```
atmosIQ/
├── docs/                     # Comprehensive documentation & architectural specs
│   ├── architecture.md       # Platform system architecture & Mermaid diagrams
│   ├── branching_strategy.md # GitFlow branch naming & conventional commits guide
│   ├── setup_guide.md        # Local environment setup guide
│   ├── data_pipeline.md      # Phase 1 Data Engineering specifications
│   ├── feature_analysis.md   # Feature domain science & column breakdown
│   ├── feature_scaling_guide.md# Preprocessing & scaling recommendations
│   ├── feature_quality_report.md# Feature quality matrix & distribution stats
│   ├── feature_leakage_report.md# Anti-leakage audit verification report
│   └── phase2_complete_guide.md# Comprehensive Phase 2 Feature Engineering guide
├── docker/                   # Docker service configs and DB initialization
│   ├── mlflow/               # MLflow server Dockerfile with PostgreSQL driver
│   └── postgres/             # init.sql script with pgvector extension setup
├── ml/                       # Machine Learning subsystem
│   ├── data/                 # Data directory structure
│   │   ├── raw/              # Immutable raw ingested datasets (openaq, firms, meteo, calendar)
│   │   ├── processed/        # Master dataset & engineered feature matrix (feature_dataset.csv)
│   │   └── logs/             # Pipeline execution log files
│   ├── notebooks/            # Exploratory research notebooks
│   ├── src/                  # Production python package
│   │   ├── ingestion/        # OpenAQ, FIRMS, Open-Meteo, Calendar ingestion drivers
│   │   ├── preprocessing/    # Data validation suite & master merge pipeline
│   │   ├── features/         # Modular feature transformers (time, weather, fire, pollution, interaction, pipeline)
│   │   └── utils/            # Helper utilities and logger setup
│   ├── configs/              # Model & ingestion YAML configs (ingestion_config.yaml, model_config.yaml)
│   ├── tests/                # Unit test suite (test_data_pipeline.py, test_features.py)
│   └── requirements.txt      # Python dependencies
├── fastapi/                  # FastAPI inference microservice (Phase 4+)
├── spring-backend/           # Spring Boot 3.3 (Java 21) orchestration backend
│   ├── pom.xml               # Maven configuration with Spring AI, JPA, pgvector
│   └── src/
│       ├── main/java/com/atmosiq/AtmosIQApplication.java
│       └── test/
├── frontend/                 # Web dashboard UI (Phase 5+)
├── rag/                      # Policy document & vector embedding store
├── scripts/                  # Shell & Python launcher scripts
│   ├── setup_env.sh          # Virtual environment setup script
│   ├── start_infrastructure.sh# Docker compose infrastructure launcher
│   └── run_data_pipeline.sh  # Phase 1 Data Engineering runner script
├── run_feature_pipeline.py   # Phase 2 Feature Engineering launcher script
├── docker-compose.yml        # Docker compose for PostgreSQL, pgAdmin, MLflow
├── pytest.ini                # Pytest path and configuration
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # Root README
```

---

## Quick Start & Verification Workflow

### 1. Environment Setup
```bash
# Clone repository
git clone git@github.com:Suraj-codes1410/AtmosIQ.git
cd AtmosIQ

# Configure environment & start Docker infrastructure (PostgreSQL, pgAdmin, MLflow)
cp .env.example .env
./scripts/start_infrastructure.sh

# Setup Python virtual environment
./scripts/setup_env.sh
source venv/bin/activate
```

### 2. Run Data Engineering Pipeline (Phase 1)
```bash
./scripts/run_data_pipeline.sh
```
*Output:* Ingests raw data and generates [`ml/data/processed/master_dataset.csv`](file:///home/suraj/atmosIQ/ml/data/processed/master_dataset.csv).

### 3. Run Feature Engineering Pipeline (Phase 2)
```bash
python run_feature_pipeline.py
```
*Output:* Generates [`ml/data/processed/feature_dataset.csv`](file:///home/suraj/atmosIQ/ml/data/processed/feature_dataset.csv) (**731 daily rows × 256 engineered features**).

### 4. Execute Unit Test Suite
```bash
pytest ml/tests/
```
*Output:* **`8 passed`** (Validates data ingestion, schema validation, wind math, lag/rolling windows, and zero target leakage).

---

## Technical Documentation Sitemap

- [Architecture Specification](file:///home/suraj/atmosIQ/docs/architecture.md)
- [Branching Strategy & Git Workflow](file:///home/suraj/atmosIQ/docs/branching_strategy.md)
- [Developer Setup Guide](file:///home/suraj/atmosIQ/docs/setup_guide.md)
- [Phase 1 Data Engineering Spec](file:///home/suraj/atmosIQ/docs/data_pipeline.md)
- [Phase 2 Complete Feature Guide](file:///home/suraj/atmosIQ/docs/phase2_complete_guide.md)
- [Feature Domain Analysis](file:///home/suraj/atmosIQ/docs/feature_analysis.md)
- [Feature Scaling & Preprocessing Recommendations](file:///home/suraj/atmosIQ/docs/feature_scaling_guide.md)
- [Feature Quality Matrix Report](file:///home/suraj/atmosIQ/docs/feature_quality_report.md)
- [Feature Anti-Leakage Audit Report](file:///home/suraj/atmosIQ/docs/feature_leakage_report.md)
