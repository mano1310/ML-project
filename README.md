# ChurnLens

ChurnLens is an end-to-end telecom customer churn propensity project. It joins subscriber, usage, recharge, and complaints signals, applies a business-driven inactivity label, and gives retention stakeholders a focused GitHub Pages view instead of a raw probability dump.

## What is included

- Deterministic telecom demo data representing subscriber, usage, recharge, and complaint tables for a zero-credential quick start.
- Behavioral feature engineering across tenure, revenue, usage intensity, recharge recency, failed payments, and complaints.
- Tuned Logistic Regression and Random Forest candidates with cross-validation, ROC AUC, average precision, Brier score, and KS evaluation.
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

Replace `make_synthetic_telecom_data` with governed subscriber, usage, recharge, and complaints extracts joined at the subscriber and observation-window grain. Define the production label as no recharge or usage for 30 consecutive days after the observation window, persist the model and metrics in an artifact registry, add data drift and calibration monitoring, and require human review for retention actions. The application intentionally frames predictions as prioritization signals rather than automatic customer decisions.

## Deploy with GitHub

Push this repository to GitHub and enable GitHub Pages with GitHub Actions as the source. The included `pages.yml` workflow publishes the `docs` dashboard on every push to `main`; the API can be deployed separately using the included Render or Docker configuration. The Pages dashboard is self-contained and does not require a backend or secrets.
