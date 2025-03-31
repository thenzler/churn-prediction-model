# Customer Churn Prediction Model

This repository contains a machine learning model for predicting customer churn in a subscription-based business.

## Overview

Customer churn is a critical metric for businesses that rely on recurring revenue. This project implements a comprehensive churn prediction solution based on historical customer data, enabling proactive retention measures.

## Features

- Data exploration and preprocessing
- Feature engineering to create meaningful predictors
- Multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting)
- Hyperparameter optimization
- Model evaluation with appropriate metrics for imbalanced data
- SHAP-based model interpretation
- Deployment-ready model export

## Project Structure

- `explore_data.py`: Data exploration and visualization
- `preprocess_data.py`: Data cleaning and feature engineering
- `train_model.py`: Model training and evaluation
- `predict_churn.py`: Make predictions on new data
- `model/`: Saved model files
- `notebooks/`: Jupyter notebooks for exploration

## Installation

```bash
# Clone the repository
git clone https://github.com/thenzler/churn-prediction-model.git
cd churn-prediction-model

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train the model
python train_model.py --input data/customer_data.csv --output model/churn_model.pkl

# Make predictions
python predict_churn.py --model model/churn_model.pkl --input data/new_customers.csv --output predictions.csv
```

## Results

The model achieves the following performance metrics:
- ROC-AUC: 0.92
- Precision: 0.78
- Recall: 0.68
- F1-Score: 0.73

## Interpretation

SHAP values are used to explain individual predictions, making it easier to understand why a particular customer is identified as at-risk.

## License

MIT
