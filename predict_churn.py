#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Churn prediction on new customer data.
This module loads a trained model and makes predictions on new customer data.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
import shap
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load a trained churn prediction model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict_churn(model, data):
    """
    Make predictions on customer data
    
    Args:
        model: Trained churn prediction model
        data: DataFrame with customer data
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Making predictions on {len(data)} customers")
    
    # Make a copy of the input data
    results = data.copy()
    
    try:
        # Make predictions
        churn_predictions = model.predict(data)
        churn_probabilities = model.predict_proba(data)[:, 1]
        
        # Add predictions to results
        results['churn_prediction'] = churn_predictions
        results['churn_probability'] = churn_probabilities
        
        # Add risk category
        results['risk_category'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        logger.info("Predictions completed")
        logger.info(f"Risk distribution: {results['risk_category'].value_counts()}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def explain_predictions(model, data, num_explanations=5):
    """
    Generate SHAP explanations for predictions
    
    Args:
        model: Trained churn prediction model
        data: DataFrame with customer data
        num_explanations: Number of individual explanations to generate
        
    Returns:
        Dictionary with explanation information
    """
    logger.info(f"Generating explanations for {num_explanations} customers")
    
    try:
        # Extract the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Get preprocessed data
        if 'preprocessor' in model.named_steps:
            X_processed = model.named_steps['preprocessor'].transform(data)
        else:
            X_processed = data
        
        # Create explainer based on model type
        if hasattr(classifier, 'estimators_'):  # Tree-based models like RandomForest
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_processed)
            
            # For random forest, shap_values is a list of arrays (one per class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get values for positive class
        else:
            # For other models, use KernelExplainer
            explainer = shap.KernelExplainer(
                classifier.predict_proba, 
                shap.kmeans(X_processed, 50)
            )
            shap_values = explainer.shap_values(X_processed)[1]  # For positive class
        
        # Get feature names
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif 'preprocessor' in model.named_steps and hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        # Create directory for visualizations
        os.makedirs('visualizations', exist_ok=True)
        
        # Generate individual explanations
        explanations = []
        
        # Identify the highest risk customers
        if 'churn_probability' in data.columns:
            high_risk_indices = data['churn_probability'].sort_values(ascending=False).index[:num_explanations]
        else:
            probabilities = model.predict_proba(data)[:, 1]
            high_risk_indices = np.argsort(-probabilities)[:num_explanations]
        
        # Generate explanations for selected customers
        for i, idx in enumerate(high_risk_indices):
            # Get the row index in the processed data
            if hasattr(data, 'iloc'):
                row_idx = data.index.get_loc(idx)
            else:
                row_idx = idx
            
            # Get customer ID or create a substitute
            if 'customer_id' in data.columns:
                customer_id = data.iloc[row_idx]['customer_id']
            else:
                customer_id = f"customer_{idx}"
            
            # Generate the plot
            plt.figure(figsize=(12, 6))
            
            # For tree-based models
            if hasattr(classifier, 'estimators_'):
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                shap.force_plot(
                    expected_value,
                    shap_values[row_idx],
                    X_processed[row_idx],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
            else:
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[row_idx],
                    X_processed[row_idx],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
            
            plt.title(f"Explanation for {customer_id}")
            plt.tight_layout()
            explanation_path = f"visualizations/explanation_{customer_id}.png"
            plt.savefig(explanation_path)
            plt.close()
            
            # Get top factors
            feature_importance = np.abs(shap_values[row_idx])
            top_indices = np.argsort(-feature_importance)[:5]  # Get top 5 features
            top_factors = [{'feature': feature_names[i], 'importance': float(feature_importance[i])} for i in top_indices]
            
            # Store explanation
            explanations.append({
                'customer_id': customer_id,
                'risk_probability': float(probabilities[row_idx]) if 'probabilities' in locals() else float(data.iloc[row_idx]['churn_probability']),
                'top_factors': top_factors,
                'explanation_path': explanation_path
            })
        
        logger.info(f"Generated {len(explanations)} explanations")
        return explanations
    
    except Exception as e:
        logger.error(f"Error generating explanations: {e}")
        logger.exception(e)
        return []

def main():
    """
    Main function for predicting churn on new customer data
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict churn on new customer data')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--input', required=True, help='Path to input CSV file with customer data')
    parser.add_argument('--output', required=True, help='Path to save predictions')
    parser.add_argument('--explain', action='store_true', help='Generate explanations for predictions')
    parser.add_argument('--num-explanations', type=int, default=5, help='Number of explanations to generate')
    
    args = parser.parse_args()
    
    logger.info("Starting churn prediction")
    
    # Load the model
    model = load_model(args.model)
    
    # Load customer data
    logger.info(f"Loading customer data from {args.input}")
    try:
        data = pd.read_csv(args.input)
        logger.info(f"Loaded {len(data)} customer records")
    except Exception as e:
        logger.error(f"Error loading customer data: {e}")
        raise
    
    # Make predictions
    results = predict_churn(model, data)
    
    # Save predictions
    logger.info(f"Saving predictions to {args.output}")
    results.to_csv(args.output, index=False)
    
    # Generate explanations if requested
    if args.explain:
        explanations = explain_predictions(model, data, args.num_explanations)
        
        # Save explanations as JSON
        explanation_path = f"{os.path.splitext(args.output)[0]}_explanations.json"
        with open(explanation_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        logger.info(f"Explanations saved to {explanation_path}")
    
    # Print summary
    risk_counts = results['risk_category'].value_counts()
    logger.info("Prediction summary:")
    for category, count in risk_counts.items():
        logger.info(f"  {category}: {count} customers ({count/len(results)*100:.1f}%)")
    
    # High-risk customers summary
    if 'customer_id' in results.columns:
        high_risk = results[results['risk_category'] == 'High Risk']
        if len(high_risk) > 0:
            logger.info("Top 5 highest churn risk customers:")
            top_risk = high_risk.sort_values('churn_probability', ascending=False).head(5)
            for _, row in top_risk.iterrows():
                logger.info(f"  {row['customer_id']}: {row['churn_probability']:.2f} probability")
    
    logger.info("Prediction completed successfully")

if __name__ == "__main__":
    main()
