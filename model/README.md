# Model Directory

This directory contains the saved machine learning models for churn prediction.

Models are saved in pickle format with the following naming convention:
- `churn_model_YYYYMMDD_HHMMSS.pkl`: Versioned model files
- `churn_model_latest.pkl`: Symlink to the most recently trained model

Model metadata is stored in `model_metadata.json`.
