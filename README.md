# Air Quality Prediction MLOps Pipeline

## Project Overview

This project is an end-to-end MLOps pipeline for predicting air quality levels using the UCI Air Quality dataset.

The system includes:

* Data preprocessing
* Model training
* Experiment tracking with MLflow
* Model serving with FastAPI
* Monitoring with Prometheus and Grafana
* Drift reporting with Evidently AI
* Interactive dashboards with Streamlit
* Data and artifact versioning with DVC
* CI/CD with GitHub Actions
* Docker-based deployment

The target variable for prediction is:

* `CO(GT)` — Carbon Monoxide concentration.

---

# Requirements

Before running the project on a new device, install:

* Git
* Python 3.10
* pip
* Docker Desktop

Verify installation:

```bash
git --version
python --version
pip --version
docker --version
```

---

# Setup on a New Device

## 1. Clone the Repository

```bash
git clone https://github.com/aliaGuda/air-quality.git
cd air-quality
```

---

## 2. Create and Activate Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[s3]"
```

---

## 4. Configure DVC Remote

The project uses Cloudflare R2 as the DVC remote storage.

Check the configured remote:

```bash
dvc remote list
```

Expected remote:

```text
myremote    s3://airquality/dvcstore
```

Configure the Cloudflare R2 endpoint:

```bash
dvc remote modify myremote endpointurl https://d16a42901b34328b626d5d537f92e0b0.r2.cloudflarestorage.com
dvc remote modify myremote region auto
dvc remote modify myremote listobjects false
```

Set credentials locally:

```bash
dvc remote modify --local myremote access_key_id YOUR_KEY
dvc remote modify --local myremote secret_access_key YOUR_SECRET
```

Pull data, models, MLflow files, and artifacts:

```bash
dvc pull -r myremote
```

---

# Running the Full System

Start all Docker services:

```bash
docker compose up --build
```

After the containers start, open:

| Service | URL |
|---|---|
| FastAPI API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5002 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

# Running Streamlit Dashboards

The project includes two Streamlit dashboards.

Run the monitoring dashboard:

```bash
streamlit run dashboard/monitoring_dashboard.py
```

Run the Streamlit app dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

---

# Viewing Evidently Reports

The Evidently HTML reports are generated in:

```text
src\monitoring\evidently_reports\
```

Open the baseline report:

```powershell
start src\monitoring\evidently_reports\baseline_report.html
```

Open the drift report:

```powershell
start src\monitoring\evidently_reports\drift_report.html
```

If the reports are not generated yet, run:

```bash
python src/monitoring/evidently_reports/baseline_report.py
python src/monitoring/evidently_reports/drift_report.py
```

Then open the HTML files again.

---

# MLflow Persistence

MLflow data and artifacts are persisted locally so they remain available after Docker rebuilds.

The project uses:

```text
MLflow database/artifacts metadata → mlflow_data/
MLflow model artifacts             → mlartifacts/
```

These folders should exist after pulling DVC artifacts:

```text
mlflow_data/
mlartifacts/
```

This allows the API, MLflow UI, and saved experiment/model artifacts to keep working after rebuilding containers.

---

# Repository Structure

```text
air-quality/
│
├── .github/workflows/
│   └── ci.yml
│
├── configs/
│   └── params.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── docs/
│   ├── model_card.md
│   └── data_card.md
│
├── dashboard/
│   ├── monitoring_dashboard.py
│   └── streamlit_app.py
│
├── monitoring/
│   ├── evidently_reports/
│   └── grafana/
│
├── models/
│   ├── model.joblib
│   └── preprocessing_pipeline.joblib
│
├── mlflow_data/
├── mlartifacts/
│
├── src/
│   ├── training/
│   ├── preprocessing/
│   ├── serving/
│   └── monitoring/
│
├── validation/
│   └── model_validation.py
│
├── tests/
│   └── unit/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Example Prediction Request

Use this JSON body in Swagger UI at:

```text
http://localhost:8000/docs
```

Endpoint:

```text
POST /predict
```

Request body:

```json
{
  "features": {
    "PT08.S1(CO)": 1360,
    "NMHC(GT)": 150,
    "C6H6(GT)": 11.9,
    "PT08.S2(NMHC)": 1046,
    "NOx(GT)": 166,
    "PT08.S3(NOx)": 1056,
    "NO2(GT)": 113,
    "PT08.S4(NO2)": 1692,
    "PT08.S5(O3)": 1268,
    "T": 13.6,
    "RH": 48.9,
    "AH": 0.7578,
    "month": 3,
    "day": 10,
    "day_of_week": 1,
    "hour": 14
  }
}
```

---

# Example Batch Prediction Request

Endpoint:

```text
POST /predict/batch
```

Request body:

```json
{
  "records": [
    {
      "features": {
        "PT08.S1(CO)": 1360,
        "NMHC(GT)": 150,
        "C6H6(GT)": 11.9,
        "PT08.S2(NMHC)": 1046,
        "NOx(GT)": 166,
        "PT08.S3(NOx)": 1056,
        "NO2(GT)": 113,
        "PT08.S4(NO2)": 1692,
        "PT08.S5(O3)": 1268,
        "T": 13.6,
        "RH": 48.9,
        "AH": 0.7578,
        "month": 3,
        "day": 10,
        "day_of_week": 1,
        "hour": 14
      }
    },
    {
      "features": {
        "PT08.S1(CO)": 1200,
        "NMHC(GT)": 130,
        "C6H6(GT)": 9.5,
        "PT08.S2(NMHC)": 980,
        "NOx(GT)": 140,
        "PT08.S3(NOx)": 900,
        "NO2(GT)": 95,
        "PT08.S4(NO2)": 1550,
        "PT08.S5(O3)": 1100,
        "T": 18.2,
        "RH": 52.0,
        "AH": 0.9,
        "month": 4,
        "day": 12,
        "day_of_week": 2,
        "hour": 10
      }
    }
  ]
}
```

---

# Health Check

Check that the API is running:

```text
http://localhost:8000/health
```

Expected response should show that the service is healthy and the model is loaded.

---

# Running Tests

```bash
pytest tests/unit/ -v
```

---

# Monitoring Features

The monitoring stack includes:

* FastAPI request monitoring
* Prediction request counters
* Prediction error counters
* Prediction latency metrics
* Prometheus metrics endpoint
* Grafana dashboard visualizations
* Evidently baseline and drift reports
* Streamlit dashboard views

---

# Useful Docker Commands

Stop running containers:

```bash
docker compose down
```

Stop and remove volumes:

```bash
docker compose down -v
```

Rebuild from scratch:

```bash
docker compose up --build
```

View running containers:

```bash
docker ps
```

---

# CI/CD Pipeline

GitHub Actions automatically performs:

1. Linting
2. Unit testing
3. Coverage validation
4. Data validation
5. Model validation
6. Docker image build verification

Pipeline triggers:

* Push to main branch
* Pull requests

---

# Contributors

* Alia Guda
* Noor Abdelhady
* Moemen Mohie

---

# License

This project is for educational and academic purposes.