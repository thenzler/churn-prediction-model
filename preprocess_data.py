#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing and feature engineering for churn prediction.
This module handles the data cleaning, transformation, and feature creation
to prepare the data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load customer data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def analyze_data(df):
    """
    Perform basic exploratory data analysis
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing data")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    
    # Check for data types
    dtypes = df.dtypes
    
    # Check for class distribution if target column exists
    class_distribution = None
    if 'Churn' in df.columns:
        class_distribution = df['Churn'].value_counts(normalize=True) * 100
    
    # Basic statistics for numerical columns
    numerical_stats = df.describe()
    
    results = {
        'shape': df.shape,
        'missing_values': missing_values[missing_values > 0].to_dict(),
        'missing_percentage': missing_pct[missing_pct > 0].to_dict(),
        'dtypes': dtypes.to_dict(),
        'class_distribution': class_distribution.to_dict() if class_distribution is not None else None,
        'numerical_stats': numerical_stats
    }
    
    logger.info(f"Data analysis completed")
    return results

def identify_feature_types(df, target_column='Churn'):
    """
    Identify numerical and categorical features in the dataset
    
    Args:
        df: DataFrame with customer data
        target_column: Name of the target column
        
    Returns:
        Tuple of lists (numerical_features, categorical_features)
    """
    logger.info("Identifying feature types")
    
    # Drop target column if present
    features = df.drop(target_column, axis=1) if target_column in df.columns else df
    
    # Identify numerical and categorical features
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Identified {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
    return numerical_features, categorical_features

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline with proper handling of numerical and categorical features
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer pipeline for data preprocessing
    """
    logger.info("Creating preprocessing pipeline")
    
    # Numerical features pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    logger.info("Preprocessing pipeline created")
    return preprocessor

def engineer_features(df):
    """
    Create new features from existing ones to improve predictive power
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        DataFrame with added engineered features
    """
    logger.info("Engineering features")
    
    df_new = df.copy()
    
    # Example: Create features based on common patterns in customer data
    
    # Add frequency-based features if applicable columns exist
    if 'usage_frequency' in df.columns and 'contract_duration' in df.columns:
        df_new['usage_per_month'] = df['usage_frequency'] / df['contract_duration']
    
    # Calculate ratios if applicable
    if 'revenue' in df.columns and 'cost' in df.columns:
        df_new['margin'] = df['revenue'] - df['cost']
        df_new['margin_percentage'] = (df_new['margin'] / df['revenue']) * 100
    
    # Customer tenure-based features
    if 'tenure' in df.columns:
        df_new['tenure_group'] = pd.cut(
            df['tenure'], 
            bins=[0, 6, 12, 24, 36, float('inf')],
            labels=['0-6 months', '6-12 months', '1-2 years', '2-3 years', '3+ years']
        )
    
    # Interaction features for services if these columns exist
    service_columns = [col for col in df.columns if col.startswith('service_')]
    if len(service_columns) > 1:
        df_new['total_services'] = df[service_columns].sum(axis=1)
    
    # Support and interaction features
    if 'support_calls' in df.columns and 'tenure' in df.columns:
        df_new['support_calls_per_month'] = df['support_calls'] / df['tenure']
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} new features")
    return df_new

def prepare_data_for_training(df, target_column='Churn'):
    """
    Main function to prepare data for model training
    
    Args:
        df: DataFrame with customer data
        target_column: Name of the target column
        
    Returns:
        Tuple of (X, y, preprocessor) ready for model training
    """
    logger.info(f"Preparing data for training with target column: {target_column}")
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Extract target variable
    if target_column in df_engineered.columns:
        y = df_engineered[target_column]
        X = df_engineered.drop(target_column, axis=1)
    else:
        logger.warning(f"Target column '{target_column}' not found. Treating all columns as features.")
        X = df_engineered
        y = None
    
    # Identify feature types
    numerical_features, categorical_features = identify_feature_types(
        X, target_column=None  # We've already removed the target column
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_features, categorical_features
    )
    
    logger.info("Data preparation completed")
    return X, y, preprocessor

if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess customer data for churn prediction')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to save the preprocessed data')
    parser.add_argument('--target', default='Churn', help='Name of the target column (default: Churn)')
    
    args = parser.parse_args()
    
    # Load the data
    data = load_data(args.input)
    
    # Analyze the data
    analysis = analyze_data(data)
    print("Data Analysis Summary:")
    print(f"  Shape: {analysis['shape']}")
    print(f"  Missing values: {analysis['missing_values']}")
    print(f"  Class distribution: {analysis['class_distribution']}")
    
    # Prepare data for training
    X, y, preprocessor = prepare_data_for_training(data, target_column=args.target)
    
    # Save preprocessed data and preprocessor for later use
    import joblib
    
    # Fit the preprocessor and transform the data
    X_transformed = preprocessor.fit_transform(X)
    
    # Save the preprocessor
    joblib.dump(preprocessor, f"{args.output.split('.')[0]}_preprocessor.pkl")
    
    # Convert transformed data back to DataFrame (if needed)
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    else:
        X_df = pd.DataFrame(X_transformed)
    
    # Save the data
    if y is not None:
        result = pd.concat([X_df, y.reset_index(drop=True)], axis=1)
    else:
        result = X_df
    
    result.to_csv(args.output, index=False)
    logger.info(f"Preprocessed data saved to {args.output}")
