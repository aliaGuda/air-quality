# README.md

# Air Quality Prediction MLOps Pipeline

## Project Overview

This project is an end-to-end MLOps pipeline for predicting air quality levels using the UCI Air Quality dataset. The system includes data preprocessing, model training, experiment tracking with MLflow, model serving through FastAPI, monitoring with Prometheus and Grafana, drift detection with Evidently AI, and CI/CD automation with GitHub Actions.

The target variable for prediction is:

* `CO(GT)` — Carbon Monoxide concentration.

The project demonstrates a production-style machine learning workflow with reproducibility, monitoring, versioning, containerization, and deployment-ready components.

---

# Project Architecture

```text
                +----------------+
                |  Raw Dataset   |
                +--------+-------+
                         |
                         v
              +--------------------+
              | Data Preprocessing |
              +--------------------+
                         |
                         v
                +----------------+
                | Model Training |
                +----------------+
                         |
                         v
               +------------------+
               | MLflow Tracking  |
               +------------------+
                         |
                         v
               +------------------+
               | Model Registry   |
               +------------------+
                         |
                         v
              +---------------------+
              | FastAPI Serving App |
              +---------------------+
                         |
          +--------------+--------------+
          |                             |
          v                             v
+-------------------+        +-------------------+
| Prometheus Metrics|        | Evidently Reports |
+-------------------+        +-------------------+
          |
          v
+-------------------+
| Grafana Dashboard |
+-------------------+
```

---

# Tech Stack

| Component            | Technology           |
| -------------------- | -------------------- |
| Programming Language | Python 3.10          |
| API Framework        | FastAPI              |
| Experiment Tracking  | MLflow               |
| Monitoring           | Prometheus + Grafana |
| Drift Detection      | Evidently AI         |
| CI/CD                | GitHub Actions       |
| Containerization     | Docker               |
| Data Versioning      | DVC                  |
| Testing              | Pytest               |
| Modeling             | Scikit-learn         |
| Visualization        | Streamlit            |

---

# Dataset Information

Dataset: UCI Air Quality Dataset

The dataset contains hourly averaged responses from an array of chemical sensors embedded in an air quality monitoring device.

Main Features:

* Temperature
* Relative Humidity
* Absolute Humidity
* NOx concentration
* NO2 concentration
* Sensor measurements

Target Variable:

* `CO(GT)`

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
│   └── monitoring_dashboard.py
│
├── monitoring/
│   ├── evidently_reports/
│   └── grafana/
│
├── models/
│   ├── model.joblib
│   └── preprocessing_pipeline.joblib
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

# Quickstart

The reviewer can run the entire serving application in three commands.

## 1. Clone Repository

```bash
git clone https://github.com/aliaGuda/air-quality.git
cd air-quality
```

## 2. Start Docker Services

```bash
docker compose up --build
```

## 3. Run Streamlit Dashboard

```bash
streamlit run dashboard/monitoring_dashboard.py
```

---

# Service Endpoints

| Service             | URL                                                      |
| ------------------- | -------------------------------------------------------- |
| FastAPI Serving App | [http://localhost:8000](http://localhost:8000)           |
| Swagger UI          | [http://localhost:8000/docs](http://localhost:8000/docs) |
| MLflow UI           | [http://localhost:5002](http://localhost:5002)           |
| Prometheus          | [http://localhost:9090](http://localhost:9090)           |
| Grafana             | [http://localhost:3000](http://localhost:3000)           |
| Streamlit Dashboard | [http://localhost:8501](http://localhost:8501)           |

---

# Example Prediction Request

```json
{
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
  "AH": 0.7578
}
```

---

# Running Tests

```bash
pytest tests/unit/ -v
```

---

# Monitoring Features

The monitoring pipeline includes:

* Prometheus metrics collection
* Grafana visualization dashboards
* MLflow experiment tracking
* Evidently AI data drift reports
* API latency monitoring
* Prediction logging

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

# Future Improvements

* Kubernetes deployment
* Automated retraining pipeline
* Cloud deployment
* Advanced alerting system
* Feature store integration
* Canary deployment support

---

# Contributors

* Alia Guda
* Noor Abdelhady
* Moemen Mohie

---

# License

This project is for educational and academic purposes.

---
