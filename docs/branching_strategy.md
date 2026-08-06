# atmosIQ Git & Branching Strategy

To maintain production stability across data engineering, machine learning, backend microservices, and frontend platforms, atmosIQ follows a structured GitFlow / Feature-Branch workflow.

## Branch Naming Convention

All branches must strictly adhere to the following prefix format:

- `main` / `master`: Production-ready code only. Tags correspond to stable release versions (e.g. `v0.1.0`).
- `develop`: Integration branch for upcoming release cycles.
- `feature/<component>/<short-description>`: New feature implementations.
  - Example: `feature/ml/ingestion-cpcb-api`
  - Example: `feature/spring/pgvector-store`
  - Example: `feature/fastapi/shap-endpoint`
- `bugfix/<component>/<issue-id>-<description>`: Non-urgent bug fixes.
  - Example: `bugfix/ml/fix-missing-imputer-leak`
- `hotfix/<description>`: Critical production patches branched directly from `main`.
- `release/vX.Y.Z`: Release candidate preparation branch.

## Commit Message Convention (Conventional Commits)

Commit messages must follow the standard [Conventional Commits specification](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

### Commit Types
- `feat`: A new feature added to the platform.
- `fix`: A bug fix.
- `docs`: Documentation changes only.
- `style`: Changes that do not affect code meaning (formatting, missing semi-colons, whitespace).
- `refactor`: Code change that neither fixes a bug nor adds a feature.
- `perf`: Performance improvements.
- `test`: Adding or correcting tests.
- `build`: Changes affecting build system or external dependencies (e.g. `pom.xml`, `requirements.txt`).
- `ci`: CI/CD pipeline configuration changes.
- `chore`: Maintenance tasks, repo scaffolding, tool configuration.

### Examples
- `chore(repo): initialize Phase 0 monorepo structure and Docker infrastructure`
- `feat(ml): implement XGBoost regressor for PM2.5 baseline model`
- `feat(spring): configure pgvector vector store bean with Spring AI`
- `docs(arch): update subsystem architecture diagram for SHAP pipeline`

## Pull Request Guidelines
1. All PRs must target `develop` (except hotfixes which target `main`).
2. PR titles must follow conventional commits format.
3. Code must pass automated linting and unit tests (`pytest` for Python, `mvn test` for Java).
4. Require minimum 1 approval before merging.
