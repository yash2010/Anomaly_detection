# Anomaly Detection Pipeline for Sensor Based Grinding Process

## Data availability

**The dataset used by this project is not published in this repository.** It originates from an industrial partner during my thesis and is covered by a **Non-Disclosure Agreement (NDA)**. No raw signals, FFT data, or derived arrays (`inputX_compressed.npz`, `labelX.npy`) are shared publicly - this repository contains only the pipeline code used to process and model that data.

## What this is

This is an experimental version of an unsupervised anomaly detection pipeline for vibration data from a grinding wheel process. Samples are FFT magnitude spectra, grouped by tool/process identifier (recorded per dressing cycle/date).

The pipeline:
1. Pretrains a 1D convolutional autoencoder (with U-Net-style skip connections) on the FFT spectra for reconstruction.
2. Extracts latent features and initializes cluster centers with K-Means.
3. Fine-tunes the autoencoder jointly with an IDEC-style clustering-consistency loss (KL divergence to a sharpened target distribution) plus a repulsion term to keep cluster centers well separated.
4. Uses per-cluster reconstruction-error thresholds to flag anomalies within a cluster (relative to its own group's typical behavior).

This version is wired up with **MLflow** (experiment tracking) and **Optuna** (hyperparameter search via `run_automl.py`) so that multiple runs, across different hyperparameter combinations (bottleneck dimension, learning rate, repulsion weight, number of clusters), can be logged, compared, and reproduced. It's meant for generating and comparing results across parameter settings, not as a final, single fixed configuration.

## Project structure

```
config.py / config.yaml   Dataclass-based config schema and default hyperparameters
run_automl.py              Optuna + MLflow driven hyperparameter search entrypoint
src/
  data_loader.py            Loads raw FFT data, filters low-sample-count groups, preprocesses/scales
  model.py                  Autoencoder architecture, latent extraction, cluster-center init, IDEC math
  train.py                  Fine-tuning loop (reconstruction + KL + repulsion loss)
  utils.py                  Plotting and diagnostics (reconstruction error, cluster/anomaly visualizations)
mlruns/                     MLflow tracking store (local file backend)
results_automl/             Saved models/artifacts per Optuna trial
db.sqlite3                  Optuna study storage
```

## Setup

Requires Python with the packages in `requirements.txt` (TensorFlow, scikit-learn, Optuna, MLflow, UMAP, Plotly, etc.).

```bash
pip install -r requirements.txt
```

## Running

Place `inputX_compressed.npz` and `labelX.npy` in the project root (not included — see Data availability above), then run:

```bash
python run_automl.py
```

This runs a small Optuna study (5 trials by default), logging each trial's parameters, losses, and silhouette score to MLflow, and saving each trial's trained model under `results_automl/`.

To inspect results:

```bash
mlflow ui
```

## Notes

- This is a research/experimentation pipeline, not a finished production system — it's used to generate and compare results across hyperparameter settings rather than to serve a single final model.
- Clustering is unsupervised; there is no ground-truth anomaly labeling in the source data, so results should be interpreted alongside domain knowledge of the grinding process rather than taken as validated anomaly labels.
