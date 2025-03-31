#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training for churn prediction.
This module trains and evaluates multiple models, performs hyperparameter
optimization, and saves the best model for future use.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            precision_recall_curve, auc, f1_score, accuracy_score, 
                            precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import shap
import joblib
import os
import logging
from datetime import datetime
import json

# Import the preprocessing module
from preprocess_data import load_data, prepare_data_for_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def split_data(X, y, test_size=0.2, stratify=True):
    """
    Split data into training and test sets
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        stratify: Whether to maintain class distribution in train/test splits
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data with test_size={test_size}, stratify={stratify}")
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=stratify_param
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def create_model_pipeline(model_type='RandomForest', preprocessor=None):
    """
    Create a pipeline with preprocessing and a specific model
    
    Args:
        model_type: Type of model to create ('LogisticRegression', 'RandomForest', 'GradientBoosting')
        preprocessor: Fitted preprocessor to include in the pipeline
        
    Returns:
        Scikit-learn pipeline with preprocessor and model
    """
    logger.info(f"Creating {model_type} model pipeline")
    
    # Define the model based on type
    if model_type == 'LogisticRegression':
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create the pipeline
    if preprocessor is not None:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])
    
    return pipeline

def train_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train multiple models and evaluate their performance
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        preprocessor: Fitted preprocessor
        
    Returns:
        Dictionary with trained models and evaluation results
    """
    logger.info("Training and evaluating models")
    
    # Define models to train
    model_types = ['LogisticRegression', 'RandomForest', 'GradientBoosting']
    
    results = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        # Create and train the model
        model_pipeline = create_model_pipeline(model_type, preprocessor)
        model_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Calculate precision-recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Store results
        results[model_type] = {
            'pipeline': model_pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Display results
        logger.info(f"{model_type} performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  PR AUC: {pr_auc:.4f}")
        
        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_type}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Create directory for visualizations if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(f'visualizations/confusion_matrix_{model_type}.png')
        plt.close()
        
        # ROC curve
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_type}')
        plt.legend()
        plt.savefig(f'visualizations/roc_curve_{model_type}.png')
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'PR AUC: {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_type}')
        plt.legend()
        plt.savefig(f'visualizations/pr_curve_{model_type}.png')
        plt.close()
    
    # Identify the best model based on ROC AUC
    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    logger.info(f"Best model: {best_model_name} with ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    return results, best_model_name

def optimize_hyperparameters(X_train, y_train, model_type, preprocessor):
    """
    Perform hyperparameter optimization for the selected model
    
    Args:
        X_train, y_train: Training data
        model_type: Type of model to optimize
        preprocessor: Fitted preprocessor
        
    Returns:
        Optimized model pipeline
    """
    logger.info(f"Optimizing hyperparameters for {model_type}")
    
    # Create base pipeline
    pipeline = create_model_pipeline(model_type, preprocessor)
    
    # Define hyperparameter grid based on model type
    if model_type == 'LogisticRegression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    elif model_type == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }
    elif model_type == 'GradientBoosting':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
    else:
        raise ValueError(f"Unsupported model type for optimization: {model_type}")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def explain_predictions(model, X_test, feature_names=None):
    """
    Generate SHAP explanations for model predictions
    
    Args:
        model: Trained model pipeline
        X_test: Test data features
        feature_names: List of feature names
        
    Returns:
        SHAP explainer and values
    """
    logger.info("Generating SHAP explanations")
    
    # Extract the classifier from the pipeline
    classifier = model.named_steps['classifier']
    
    # Get preprocessed data
    if 'preprocessor' in model.named_steps:
        X_processed = model.named_steps['preprocessor'].transform(X_test)
    else:
        X_processed = X_test
    
    # Create explainer based on model type
    if isinstance(classifier, RandomForestClassifier):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)
        
        # For random forest, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
    
    elif isinstance(classifier, GradientBoostingClassifier):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)
    
    else:
        # For other models, use KernelExplainer with a sample of the data
        # to keep computation manageable
        background = shap.kmeans(X_processed, 50)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(X_processed[:100])[1]  # For positive class
    
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif 'preprocessor' in model.named_steps and hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Create summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig('visualizations/shap_feature_importance.png')
    plt.close()
    
    # Create detailed summary plot
    plt.figure(figsize=(12, 16))
    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig('visualizations/shap_summary.png')
    plt.close()
    
    # Create individual explanations for a few examples
    for i in range(min(5, len(X_test))):
        plt.figure(figsize=(12, 5))
        if isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier)):
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, list) 
                else explainer.expected_value,
                shap_values[i],
                X_processed[i],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        else:
            shap.force_plot(
                explainer.expected_value,
                shap_values[i],
                X_processed[i],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        plt.tight_layout()
        plt.savefig(f'visualizations/shap_explanation_sample_{i}.png')
        plt.close()
    
    return explainer, shap_values

def save_model(model, output_dir='model'):
    """
    Save the trained model and metadata
    
    Args:
        model: Trained model pipeline
        output_dir: Directory to save the model
        
    Returns:
        Path to the saved model
    """
    logger.info(f"Saving model to {output_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model
    model_filename = f"{output_dir}/churn_model_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    
    # Create symlink to latest model
    latest_link = f"{output_dir}/churn_model_latest.pkl"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    
    # Create symlink (platform dependent)
    try:
        os.symlink(os.path.basename(model_filename), latest_link)
    except (OSError, AttributeError):
        # Fallback for Windows or if symlinks not supported
        import shutil
        shutil.copy2(model_filename, latest_link)
    
    logger.info(f"Model saved as {model_filename}")
    return model_filename

def main():
    """
    Main function to train and evaluate churn prediction models
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a churn prediction model')
    parser.add_argument('--input', required=True, help='Path to input CSV file with customer data')
    parser.add_argument('--output', default='model', help='Directory to save the model')
    parser.add_argument('--target', default='Churn', help='Name of the target column (default: Churn)')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    logger.info("Starting churn prediction model training")
    
    # Load and prepare data
    df = load_data(args.input)
    
    # Prepare data for training
    X, y, preprocessor = prepare_data_for_training(df, target_column=args.target)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and evaluate models
    model_results, best_model_name = train_models(X_train, y_train, X_test, y_test, preprocessor)
    
    # Optimize the best model if requested
    if args.optimize:
        logger.info(f"Starting hyperparameter optimization for {best_model_name}")
        optimized_model = optimize_hyperparameters(X_train, y_train, best_model_name, preprocessor)
        
        # Evaluate optimized model
        y_pred = optimized_model.predict(X_test)
        y_prob = optimized_model.predict_proba(X_test)[:, 1]
        
        logger.info("Optimized model performance:")
        logger.info(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"  ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        logger.info(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")
        
        # Use the optimized model as the final model
        final_model = optimized_model
    else:
        # Use the best model as the final model
        final_model = model_results[best_model_name]['pipeline']
    
    # Generate model explanations
    explain_predictions(final_model, X_test)
    
    # Save the model
    model_path = save_model(final_model, args.output)
    
    # Save evaluation results
    try:
        feature_names = X.columns.tolist()
    except:
        feature_names = None
    
    # Save model metadata
    metadata = {
        'model_type': best_model_name,
        'target_column': args.target,
        'performance': {
            'accuracy': float(accuracy_score(y_test, final_model.predict(X_test))),
            'roc_auc': float(roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])),
            'f1_score': float(f1_score(y_test, final_model.predict(X_test)))
        },
        'feature_names': feature_names,
        'created_at': datetime.now().isoformat(),
        'model_path': model_path
    }
    
    with open(f"{args.output}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
