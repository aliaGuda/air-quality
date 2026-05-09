# Model Card — Air Quality Prediction Model

# Model Details

| Field | Value |
|---|---|
| Model Name | Air Quality Regressor |
| Problem Type | Regression |
| Framework | Scikit-learn |
| Base Algorithm | Gradient Boosting Regressor |
| Tracking System | MLflow |
| Serving Framework | FastAPI |
| Current Stage | Production |

---

# Intended Use

This model predicts Carbon Monoxide concentration (`CO(GT)`) using environmental measurements and air-quality sensor readings.

The model is intended for:

* Air quality analysis
* Environmental monitoring research
* Educational MLOps demonstrations
* Machine learning experimentation
* Monitoring and observability demonstrations

The model is NOT intended for:

* Medical diagnosis
* Emergency response systems
* Government safety enforcement
* Critical environmental policy decisions
* Life-critical infrastructure systems

---

# Training Data

Dataset Used:

* UCI Air Quality Dataset

Dataset Characteristics:

* Hourly environmental sensor measurements
* Multivariate tabular dataset
* Time-series environmental observations
* Real-world air-quality sensor readings

Target Variable:

* `CO(GT)` — Carbon Monoxide concentration

Features Include:

* PT08 sensor measurements
* NOx concentration
* NO2 concentration
* Temperature
* Relative humidity
* Absolute humidity
* Engineered temporal features

---

# Input Schema

The model expects the following numerical input features:

```text
PT08.S1(CO)
NMHC(GT)
C6H6(GT)
PT08.S2(NMHC)
NOx(GT)
PT08.S3(NOx)
NO2(GT)
PT08.S4(NO2)
PT08.S5(O3)
T
RH
AH
month
day
day_of_week
hour
```

All features are validated through FastAPI request schemas before inference.

---

# Data Preprocessing

The preprocessing pipeline includes:

* Missing value handling
* Feature scaling
* Numerical transformations
* Feature engineering
* Temporal feature extraction
* Pipeline serialization using joblib

Engineered temporal features include:

* Month
* Day
* Day of week
* Hour

Preprocessing artifacts are stored separately from the trained model.

---

# Model Performance

## Overall Metrics

| Metric | Value |
|---|---|
| RMSE | 0.43 |
| MAE | 0.31 |
| R² Score | 0.84 |

Note:

Performance values may vary slightly depending on validation splits, random seeds, and retraining runs.

---

# Subgroup Performance

The dataset does not contain demographic or human-sensitive subgroups.

However, performance may vary across:

* Seasonal conditions
* Temperature ranges
* Humidity conditions
* Sensor behavior patterns

Potential degradation may occur during:

* Extreme environmental conditions
* Sensor drift situations
* Missing or corrupted sensor readings
* Distribution shifts outside training conditions

---

# Experiment Tracking

The project uses MLflow for experiment tracking and model management.

Tracked components include:

* Training runs
* Hyperparameters
* Validation metrics
* Model artifacts
* Production versions
* Experiment history

---

# Monitoring and Drift Detection

The production pipeline monitors:

* Prediction latency
* Input feature drift
* Prediction distribution drift
* Missing values
* API request volume
* Prediction errors
* Model-serving health

Monitoring and observability tools:

* Prometheus
* Grafana
* Evidently AI

---

# API Endpoints

| Endpoint | Purpose |
|---|---|
| `/predict` | Single prediction |
| `/predict/batch` | Batch predictions |
| `/health` | Service health check |
| `/metrics` | Prometheus monitoring metrics |

---

# Artifact Management

Versioned artifacts include:

* Processed datasets
* Trained models
* Preprocessing pipelines
* MLflow registry database
* MLflow artifacts
* Monitoring artifacts

Artifact versioning and persistence are managed using:

* DVC
* Cloudflare R2 storage
* Docker-mounted persistent volumes

---

# Infrastructure Stack

The deployment stack includes:

* Docker Compose
* FastAPI
* MLflow
* DVC
* Prometheus
* Grafana
* Streamlit
* Evidently AI
* GitHub Actions

---

# Ethical Considerations

Potential risks include:

* Incorrect predictions under unseen environmental conditions
* Sensor bias affecting outputs
* Distribution drift over time
* Misuse for high-stakes decision-making

Mitigations include:

* Continuous monitoring
* Drift detection reports
* Input validation checks
* Reproducible training pipelines
* Clear non-production disclaimers

---

# Limitations

Known limitations include:

* Limited geographic diversity in training data
* Historical dataset may not reflect current pollution patterns
* Sensitivity to missing sensor values
* Limited generalization outside training conditions
* Environmental distribution shifts over time

---

# Reproducibility

Experiment reproducibility is supported using:

* MLflow experiment tracking
* DVC artifact versioning
* Docker containerization
* Git version control
* Parameter configuration files
* Persistent MLflow storage

---

# Deployment Information

Serving Method:

* FastAPI REST API

Containerization:

* Docker Compose

Monitoring:

* Prometheus + Grafana

Drift Detection:

* Evidently AI

Experiment Tracking:

* MLflow

Dashboards:

* Streamlit

---

# Contact

Project developed as part of an academic MLOps engineering project.

---