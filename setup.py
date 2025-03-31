from setuptools import setup, find_packages

setup(
    name="churn_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.39.0",
        "joblib>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning model for customer churn prediction",
    keywords="machine learning, churn prediction, customer analytics",
    url="https://github.com/thenzler/churn-prediction-model",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "explore-data=explore_data:main",
            "preprocess-data=preprocess_data:main",
            "train-model=train_model:main",
            "predict-churn=predict_churn:main",
            "run-pipeline=run_pipeline:main",
        ],
    },
)
