# Data Card — UCI Air Quality Dataset

## Dataset Overview

| Field           | Value                           |
| --------------- | ------------------------------- |
| Dataset Name    | UCI Air Quality Dataset         |
| Source          | UCI Machine Learning Repository |
| Data Type       | Multivariate Time Series        |
| Task            | Regression                      |
| Target Variable | CO(GT)                          |

---

## Dataset Description

The dataset contains hourly averaged measurements from an array of gas sensors deployed in an Italian city.

The data was collected for air quality monitoring and environmental analysis.

Measurements include:

* Carbon Monoxide
* Nitrogen Oxides
* Benzene concentration
* Temperature
* Relative Humidity
* Absolute Humidity
* Sensor outputs

---

## Data Source

Official Source:

* UCI Machine Learning Repository

Original creators:

* Environmental monitoring researchers

The dataset is publicly available for educational and research use.

---

## Schema

### Input Features

| Feature       | Description                      |
| ------------- | -------------------------------- |
| PT08.S1(CO)   | Tin oxide sensor response for CO |
| NMHC(GT)      | Non-methane hydrocarbons         |
| C6H6(GT)      | Benzene concentration            |
| PT08.S2(NMHC) | Sensor response for NMHC         |
| NOx(GT)       | Nitrogen oxides concentration    |
| PT08.S3(NOx)  | Sensor response for NOx          |
| NO2(GT)       | Nitrogen dioxide concentration   |
| PT08.S4(NO2)  | Sensor response for NO2          |
| PT08.S5(O3)   | Sensor response for ozone        |
| T             | Temperature                      |
| RH            | Relative humidity                |
| AH            | Absolute humidity                |

### Target Variable

| Feature | Description                   |
| ------- | ----------------------------- |
| CO(GT)  | Carbon monoxide concentration |

---

## Data Preprocessing Decisions

The following preprocessing steps were applied:

1. Missing value handling
2. Invalid measurement filtering
3. Feature engineering
4. Scaling and normalization
5. Train-validation-test splitting
6. Pipeline serialization

Missing values were handled using preprocessing pipelines instead of manual deletion to preserve dataset size.

---

## Data Quality Considerations

Known issues in the dataset include:

* Missing values
* Sensor noise
* Potential outliers
* Temporal inconsistencies
* Historical environmental drift

The preprocessing pipeline was designed to reduce the impact of these issues.

---

## Bias and Fairness Notes

This dataset does not contain demographic information.

However, environmental bias may exist because:

* Data originates from a limited geographic region
* Pollution conditions may not generalize globally
* Seasonal variations may influence distributions

The model should not be assumed to generalize to all cities or climates.

---

## Privacy Considerations

The dataset contains:

* No personally identifiable information
* No sensitive user data
* No demographic information

Privacy risk is considered minimal.

---

## Licensing and Usage

Dataset usage follows the terms provided by the UCI Machine Learning Repository.

Intended usage:

* Academic projects
* Research experiments
* Educational demonstrations

Not intended for:

* Commercial environmental compliance systems
* Critical infrastructure monitoring without additional validation

---

## Versioning and Tracking

Dataset management tools:

* DVC for data versioning
* Git for code tracking
* MLflow for experiment tracking

Processed datasets are stored separately from raw data to maintain reproducibility.

---

## Data Validation

Validation checks include:

* Missing value checks
* Schema validation
* Feature consistency checks
* Distribution monitoring
* Drift detection reports

---

## Monitoring Integration

The dataset supports monitoring workflows including:

* Feature drift detection
* Prediction drift monitoring
* Performance tracking
* Input anomaly detection

Tools used:

* Evidently AI
* Prometheus
* Grafana
