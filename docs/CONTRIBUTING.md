# EDAPHNET Contributing Guidelines

Thank you for contributing to EDAPHNET. We maintain strict engineering and MLOps standards.

---

## 1. Code Standards

- **Formatting**: `black` line length 100.
- **Linting**: `ruff` without errors.
- **Typing**: Python 3.12 type hints on all functions and methods.
- **Data Contracts**: Pydantic models with explicit field descriptions and validation constraints.
- **Testing**: All PRs must maintain or improve test coverage. Run `pytest tests -v`.

---

## 2. Pull Request Workflow

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes following Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
3. Ensure all tests pass:
   ```bash
   make test
   make lint
   ```
4. Push your branch and open a Pull Request against `main`.
