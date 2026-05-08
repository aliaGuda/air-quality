# Model Card — Air Quality Prediction Model

## Model Details

| Field             | Value                 |
| ----------------- | --------------------- |
| Model Name        | Air Quality Regressor |
| Model Type        | Regression            |
| Framework         | Scikit-learn          |
| Tracking System   | MLflow                |
| Serving Framework | FastAPI               |
| Version           | Production v1         |

---

## Intended Use

This model predicts Carbon Monoxide concentration (`CO(GT)`) based on environmental and sensor measurements.

The model is intended for:

* Air quality analysis
* Environmental monitoring
* Educational MLOps demonstrations
* Research and experimentation

The model is NOT intended for:

* Medical diagnosis
* Government safety decisions
* Critical environmental policy decisions
* Life-critical systems

---

## Training Data

Dataset Used:

* UCI Air Quality Dataset

Characteristics:

* Hourly sensor measurements
* Environmental variables
* Multivariate tabular dataset

Target Variable:

* `CO(GT)`

Features Include:

* NOx concentration
* NO2 concentration
* Temperature
* Humidity
* Sensor responses

---

## Data Preprocessing

The preprocessing pipeline includes:

* Missing value handling
* Feature scaling
* Feature selection
* Numerical transformations
* Pipeline serialization using joblib

Preprocessing artifacts are stored separately from the trained model.

---

## Model Performance

### Overall Metrics

| Metric   | Value |
| -------- | ----- |
| RMSE     | 0.43  |
| MAE      | 0.31  |
| R² Score | 0.84  |

Note: Values may vary slightly depending on training runs and validation splits.

---

## Subgroup Performance

The dataset does not contain demographic or human-sensitive subgroups.

However, performance may vary across:

* Temperature ranges
* Humidity conditions
* Seasonal environmental patterns

Potential performance degradation may occur during:

* Extreme environmental conditions
* Sensor drift situations
* Missing or corrupted sensor measurements

---

## Monitoring and Drift Detection

The production pipeline monitors:

* Prediction latency
* Input feature drift
* Prediction distribution drift
* Missing values
* API request volume

Tools Used:

* Prometheus
* Grafana
* Evidently AI

---

## Ethical Considerations

Potential risks include:

* Incorrect predictions under unseen environmental conditions
* Sensor bias affecting outputs
* Misuse for high-stakes decision making

Mitigations:

* Continuous monitoring
* Drift detection reports
* Validation checks
* Clear non-production disclaimers

---

## Limitations

Known limitations:

* Limited geographic diversity in training data
* Historical dataset may not represent current pollution patterns
* Sensitive to missing sensor values
* Limited generalization outside training conditions

---

## Reproducibility

Experiment tracking and reproducibility are supported using:

* MLflow
* DVC
* Docker
* Git version control
* Parameter configuration files

---

## Deployment Information

Serving Method:

* FastAPI REST API

Containerization:

* Docker Compose

Monitoring:

* Prometheus + Grafana

Tracking:

* MLflow

---

## Contact

Project developed as part of an MLOps academic project.

---


