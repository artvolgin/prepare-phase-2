# RASKA-Team Solution - PREPARE Challenge: Phase 2

## Summary

We developed a robust, fair, and explainable pipeline for predicting cognitive scores from small, longitudinal tabular data by integrating TabPFN-derived quantile features, socio-demographic and temporal feature engineering, and ensemble modeling with blending techniques.

# Setup

1. Create an environment using Python 3.11 The solution was originally run on Python 3.11.8
```
conda create --name prepare-submission python=3.11.8
conda activate prepare-submission
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Install the project in editable mode to enable src imports
```
pip install -e .
```

4. Download the data from the competition page into `data/raw`

# Project Structure

Before running training or inference, the directory should be structured as follows:

```
├── README.md
├── data
│   ├── confact/            # Generated datasets for counterfactual estimation
│   ├── processed/          # Preprocessed datasets
│   └── raw/                # Raw data provided by organizers
├── models/                 # Trained models and their parameters
├── notebooks/
│   └── report_output.ipynb # Notebook with report figures and tables
├── output/
│   └── confact/            # Model predictions on counterfactual datasets
├── requirements.txt
├── scripts/
│   ├── create_confact.py         # Create counterfactual datasets
│   ├── run_confact.py            # Run inference on counterfactual datasets
│   ├── run_inference.py          # Run inference on new test data
│   ├── run_specifications.py     # Train and infer with multiple model/data configurations
│   └── run_train_inference.py    # Train and infer for the main specification
├── setup.py                # Installation script for editable mode
└── src/
    ├── data_builder.py          # Merge multiple input datasets
    ├── data_preprocessor.py     # Initial preprocessing
    ├── difference_predictor.py  # Predict differences between waves
    ├── ensemble_predictor.py    # Ensemble prediction logic
    ├── pipeline_runner.py       # Orchestrates the full pipeline
    └── tabpfn_runner.py         # Predict quantiles using TabPFN
```

# Hardware

The solution was run on macOS Sonoma 14.5.
-	CPU/GPU: Apple M3 Max
-	Memory: 64 GB

Both training and inference were run on CPU.
- Training time: ~30 minutes
- Inference time: ~10 minutes

# Running the Code

Main script for training and inference: `python scripts/run_train_inference.py`

Train and infer across multiple model/data specs: `python scripts/run_specifications.py`

Run inference on counterfactual datasets: `python scripts/run_specifications.py`

Run inference on new test data: `python scripts/run_inference.py`

You can download the trained models, predictions, and preprocessed datasets from the `data`, `models`, and `output` folders using the following Google Drive link: https://drive.google.com/drive/folders/1b81I9KfZ0OqDw_5zCmxOakA909v8xoK8?usp=drive_link
