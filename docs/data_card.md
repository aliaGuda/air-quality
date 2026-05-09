# Data Card — UCI Air Quality Dataset

# Dataset Overview

| Field | Value |
|---|---|
| Dataset Name | UCI Air Quality Dataset |
| Source | UCI Machine Learning Repository |
| Data Type | Multivariate Time Series |
| Task Type | Regression |
| Target Variable | `CO(GT)` |
| File Format | CSV |
| Domain | Environmental Monitoring |
| Primary Use | Air Quality Prediction |

---

# Dataset Description

The dataset contains hourly averaged environmental measurements collected from an array of gas sensors deployed in an Italian city.

The dataset was designed for:

* Air quality monitoring
* Environmental analysis
* Pollution prediction research
* Sensor-based forecasting experiments

Measurements include:

* Carbon Monoxide concentration
* Nitrogen Oxides concentration
* Benzene concentration
* Temperature
* Relative Humidity
* Absolute Humidity
* Gas sensor responses

The dataset represents real-world environmental sensor behavior and includes naturally occurring missing values and sensor noise.

---

# Data Source

Official Source:

* UCI Machine Learning Repository

Original dataset creators:

* Environmental monitoring researchers

Dataset availability:

* Publicly available for academic and research usage

---

# Dataset Characteristics

| Characteristic | Value |
|---|---|
| Dataset Type | Tabular Time-Series |
| Observation Frequency | Hourly |
| Feature Types | Numerical |
| Missing Values | Present |
| Sensor Measurements | Included |
| Engineered Features | Included after preprocessing |

---

# Input Schema

## Raw Input Features

| Feature | Description |
|---|---|
| PT08.S1(CO) | Tin oxide sensor response for CO |
| NMHC(GT) | Non-methane hydrocarbons concentration |
| C6H6(GT) | Benzene concentration |
| PT08.S2(NMHC) | Sensor response for NMHC |
| NOx(GT) | Nitrogen oxides concentration |
| PT08.S3(NOx) | Sensor response for NOx |
| NO2(GT) | Nitrogen dioxide concentration |
| PT08.S4(NO2) | Sensor response for NO2 |
| PT08.S5(O3) | Sensor response for ozone |
| T | Temperature |
| RH | Relative humidity |
| AH | Absolute humidity |

---

# Engineered Features

The preprocessing pipeline generates additional temporal features:

| Feature | Description |
|---|---|
| month | Month extracted from timestamp |
| day | Day extracted from timestamp |
| day_of_week | Day of week extracted from timestamp |
| hour | Hour extracted from timestamp |

---

# Target Variable

| Feature | Description |
|---|---|
| CO(GT) | Carbon Monoxide concentration |

---

# Data Preprocessing Pipeline

The following preprocessing operations were applied:

1. Missing value handling
2. Invalid measurement filtering
3. Temporal feature engineering
4. Numerical scaling
5. Train-validation-test splitting
6. Preprocessing pipeline serialization

Preprocessing was implemented using reusable Scikit-learn pipelines.

Missing values were handled through pipeline-based transformations rather than manual row deletion to preserve dataset size and maintain reproducibility.

---
# Dataset Splits

The dataset was divided into the following subsets:

* Training dataset
* Test dataset
* Reference dataset
* Production dataset

Purpose of each split:

| Split | Purpose |
|---|---|
| Train | Model training and preprocessing fitting |
| Test | Offline model evaluation |
| Reference | Baseline monitoring and drift comparison |
| Production | Simulated production inference monitoring |

The project uses structured/tabular regression rather than sequential forecasting.

Predictions are generated from independent environmental and sensor measurements without using:

* Lag features
* Sliding windows
* Autoregressive forecasting
* Sequential prediction models

Temporal features such as:

* month
* day
* day_of_week
* hour

are included as engineered structured features rather than sequence inputs.

The split strategy was designed to:

* Prevent data leakage
* Support reproducible experimentation
* Enable monitoring workflows
* Support drift detection using Evidently AI

The reference and production datasets are used for:

* Feature drift detection
* Prediction drift monitoring
* Data quality comparisons
* Monitoring demonstrations

---

# Data Quality Considerations

Known dataset issues include:

* Missing values
* Sensor noise
* Outliers
* Temporal inconsistencies
* Environmental drift
* Sensor degradation over time

The preprocessing pipeline was designed to reduce the impact of these issues through validation and transformation steps.

---

# Bias and Fairness Notes

The dataset does not contain:

* Personally identifiable information
* Demographic information
* Human-sensitive attributes

However, environmental bias may exist because:

* Data originates from a limited geographic region
* Pollution patterns may differ globally
* Seasonal variations affect feature distributions
* Sensor calibration may vary over time

The resulting model should not be assumed to generalize across all cities, climates, or environmental conditions without additional validation.

---

# Privacy Considerations

The dataset contains:

* No personal user data
* No demographic information
* No sensitive information
* No identifiable records

Privacy risk is considered minimal.

---

# Validation and Quality Checks

Validation checks include:

* Missing value validation
* Schema validation
* Numerical range validation
* Feature consistency checks
* Drift detection monitoring
* Distribution comparison checks

Validation is integrated into both training and serving workflows.

---

# Monitoring Integration

The dataset supports monitoring workflows including:

* Feature drift detection
* Prediction drift monitoring
* Input anomaly detection
* Data quality monitoring
* Performance tracking

Monitoring tools used:

* Evidently AI
* Prometheus
* Grafana

---

# Dataset Versioning and Tracking

Dataset and artifact management tools:

* DVC for dataset versioning
* Git for code tracking
* MLflow for experiment tracking

Tracked artifacts include:

* Raw datasets
* Processed datasets
* Training splits
* Preprocessing pipelines
* Modeling artifacts

Processed datasets are stored separately from raw datasets to preserve reproducibility and lineage tracking.

---

# Storage and Reproducibility

Dataset reproducibility is supported using:

* DVC artifact versioning
* Cloudflare R2 remote storage
* Docker-based execution environments
* Parameter configuration files
* Git-based source control

---

# Intended Usage

Recommended usage:

* Academic research
* Educational MLOps projects
* Environmental machine learning experiments
* Monitoring and observability demonstrations

---

# Non-Intended Usage

The dataset and resulting models are NOT intended for:

* Critical infrastructure systems
* Government environmental enforcement
* Emergency response systems
* Production environmental compliance systems without additional validation

---

# Contact

Dataset documentation prepared as part of an academic MLOps engineering project.

---