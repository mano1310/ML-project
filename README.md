# ChurnLens

ChurnLens is an end-to-end, explainability-first customer churn decisioning project. It trains a model, serves risk and local drivers through FastAPI, and gives retention stakeholders a focused Streamlit view instead of a raw probability dump.

## What is included

- Deterministic synthetic customer data for a zero-credential quick start, with an obvious seam for replacing it with governed production data.
- Reproducible scikit-learn pipeline: imputation, scaling, one-hot encoding, class-balanced logistic regression, and holdout metrics.
- Local explanations based on standardized model contributions relative to a reference population.
- FastAPI endpoint with schema validation and health check.
- Static GitHub Pages stakeholder tool that pairs risk with plain-language action and evidence.
- Unit/API tests, Ruff linting, Docker Compose, and GitHub Actions CI.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
churnlens-train
uvicorn churnlens.api:app --reload
```

Open `docs/index.html` directly for the static stakeholder experience, or serve the repository with `python -m http.server 8080 --directory docs`. The API is available at `http://localhost:8000/docs` when running `uvicorn churnlens.api:app --reload`.

## Production path

Replace `make_synthetic_customers` with a versioned feature view, persist the model and metrics in an artifact registry, add data drift and calibration monitoring, and require human review for retention actions. The application intentionally frames predictions as prioritization signals rather than automatic customer decisions.

## Deploy with GitHub

Push this repository to GitHub and enable GitHub Pages with GitHub Actions as the source. The included `pages.yml` workflow publishes the `docs` dashboard on every push to `main`; the API can be deployed separately using the included Render or Docker configuration. The Pages dashboard is self-contained and does not require a backend or secrets.