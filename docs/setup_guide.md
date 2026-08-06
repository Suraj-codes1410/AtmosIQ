# atmosIQ Setup & Development Guide (Phase 0)

This guide walks through setting up your local environment for the **atmosIQ** platform.

## Prerequisites

Ensure your host environment meets the following software requirements:
- **OS**: Linux / macOS / WSL2
- **Docker**: Version 20.10+ & Docker Compose v2+
- **Python**: Version 3.10+ (Python 3.14 compatible)
- **Java**: Java 21 JDK (OpenJDK 21+)
- **Maven**: Apache Maven 3.9+
- **Git**: Git 2.30+

---

## 1. Quick Start

```bash
# 1. Clone the repository
git clone <repository-url> atmosIQ
cd atmosIQ

# 2. Configure Environment Variables
cp .env.example .env

# 3. Launch Docker Infrastructure (PostgreSQL + pgvector, pgAdmin, MLflow)
./scripts/start_infrastructure.sh

# 4. Setup Python Virtual Environment
./scripts/setup_env.sh
source venv/bin/activate

# 5. Verify Spring Boot Java Backend Setup
cd spring-backend
mvn compile
mvn test
```

---

## 2. Infrastructure Services

Once running via `./scripts/start_infrastructure.sh`, access the following interfaces:

| Service | URL | Credentials / Notes |
|---|---|---|
| **PostgreSQL + pgvector** | `localhost:5432` | DB: `atmosiq_db`, User: `postgres`, Pass: `postgres` |
| **pgAdmin 4** | `http://localhost:5050` | User: `admin@atmosiq.com`, Pass: `admin` |
| **MLflow Server** | `http://localhost:5000` | SQLite/Postgres backend store |

To check container health:
```bash
docker compose ps
```

To stop infrastructure:
```bash
docker compose down
```

---

## 3. Python ML Subsystem Testing

Run sanity check unit tests:
```bash
source venv/bin/activate
pytest ml/tests/
```

---

## 4. Spring Boot Backend Setup

Build and run Spring Boot backend:
```bash
cd spring-backend
mvn spring-boot:run
```
Actuator endpoints: `http://localhost:8080/actuator/health`
