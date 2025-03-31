#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data exploration for churn prediction.
This module provides functions for exploring and visualizing customer data
to gain insights before building the prediction model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
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

def data_summary(df, target_column='Churn'):
    """
    Generate a summary of the dataset
    
    Args:
        df: DataFrame to summarize
        target_column: Name of the target variable
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating data summary")
    
    # Basic information
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Target variable distribution if it exists
    if target_column in df.columns:
        summary['target_distribution'] = df[target_column].value_counts().to_dict()
        summary['target_percentage'] = (df[target_column].value_counts(normalize=True) * 100).to_dict()
    
    # Numerical statistics
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    summary['numerical_stats'] = df[numerical_cols].describe().to_dict()
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_stats = {}
    for col in categorical_cols:
        categorical_stats[col] = {
            'unique_values': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    summary['categorical_stats'] = categorical_stats
    
    logger.info("Data summary generated")
    return summary

def identify_potential_issues(df):
    """
    Identify potential data quality issues
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with identified issues
    """
    logger.info("Identifying potential data issues")
    
    issues = {
        'missing_values': {},
        'outliers': {},
        'skewed_columns': {},
        'imbalanced_categories': {},
        'constant_columns': []
    }
    
    # Check for missing values
    missing = df.isnull().sum() / len(df) * 100
    issues['missing_values'] = missing[missing > 0].to_dict()
    
    # Check for columns with very few unique values (potentially constant)
    nunique = df.nunique() / len(df) * 100
    issues['constant_columns'] = nunique[nunique < 0.1].index.tolist()
    
    # Check for outliers in numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            issues['outliers'][col] = {
                'count': int(outliers),
                'percentage': float(outliers / len(df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
    
    # Check for highly skewed numerical columns
    for col in numerical_cols:
        skewness = stats.skew(df[col].dropna())
        if abs(skewness) > 1.0:  # Highly skewed
            issues['skewed_columns'][col] = float(skewness)
    
    # Check for imbalanced categories in categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True) * 100
        if value_counts.iloc[0] > 90:  # Dominant category > 90%
            issues['imbalanced_categories'][col] = {
                'dominant_category': value_counts.index[0],
                'percentage': float(value_counts.iloc[0])
            }
    
    logger.info("Data issues identification completed")
    return issues

def plot_target_distribution(df, target_column='Churn', output_dir='visualizations'):
    """
    Plot the distribution of the target variable
    
    Args:
        df: DataFrame with customer data
        target_column: Name of the target column
        output_dir: Directory to save visualizations
    """
    if target_column not in df.columns:
        logger.warning(f"Target column '{target_column}' not found in data")
        return
    
    logger.info(f"Plotting distribution of target variable: {target_column}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=target_column, data=df)
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for i, p in enumerate(ax.patches):
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, 
                   (p.get_x() + p.get_width() / 2., p.get_height() / 2), 
                   ha = 'center', va = 'center', fontsize=12, color='white')
    
    plt.title(f'Distribution of {target_column}', fontsize=14)
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/target_distribution.png')
    plt.close()
    
    logger.info(f"Target distribution plot saved to {output_dir}/target_distribution.png")

def plot_numerical_features(df, target_column='Churn', output_dir='visualizations', max_features=10):
    """
    Create visualizations for numerical features
    
    Args:
        df: DataFrame with customer data
        target_column: Name of the target column
        output_dir: Directory to save visualizations
        max_features: Maximum number of features to plot
    """
    logger.info("Plotting numerical features")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Exclude target if it's numerical
    if target_column in numerical_cols:
        numerical_cols = numerical_cols.drop(target_column)
    
    # Limit to max_features
    if len(numerical_cols) > max_features:
        logger.info(f"Limiting plots to {max_features} numerical features")
        # Prioritize features with highest correlation to target (if target exists)
        if target_column in df.columns and df[target_column].dtype in ('int64', 'float64', 'bool'):
            correlations = df[numerical_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
            numerical_cols = correlations.index[:max_features]
        else:
            numerical_cols = numerical_cols[:max_features]
    
    # Plot histograms with KDE
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        
        # Main plot
        if target_column in df.columns and df[target_column].nunique() <= 5:
            # Plot with target class if available
            sns.histplot(data=df, x=col, hue=target_column, kde=True, element="step")
            plt.title(f'Distribution of {col} by {target_column}', fontsize=14)
        else:
            # Basic distribution plot
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}', fontsize=14)
        
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_{col}.png')
        plt.close()
        
        # Box plot by target if categorical
        if target_column in df.columns and df[target_column].nunique() <= 5:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_column, y=col, data=df)
            plt.title(f'Distribution of {col} by {target_column}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/boxplot_{col}_by_{target_column}.png')
            plt.close()
    
    logger.info(f"Numerical feature plots saved to {output_dir}")

def plot_categorical_features(df, target_column='Churn', output_dir='visualizations', max_features=10):
    """
    Create visualizations for categorical features
    
    Args:
        df: DataFrame with customer data
        target_column: Name of the target column
        output_dir: Directory to save visualizations
        max_features: Maximum number of features to plot
    """
    logger.info("Plotting categorical features")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Exclude target if it's categorical
    if target_column in categorical_cols:
        categorical_cols = categorical_cols.drop(target_column)
    
    # Limit to max_features
    if len(categorical_cols) > max_features:
        logger.info(f"Limiting plots to {max_features} categorical features")
        # Select features with the fewest categories
        n_categories = df[categorical_cols].nunique().sort_values()
        categorical_cols = n_categories.index[:max_features]
    
    # Plot count plots
    for col in categorical_cols:
        # Skip columns with too many categories
        if df[col].nunique() > 15:
            top_categories = df[col].value_counts().head(15).index
            df_plot = df.copy()
            df_plot[col] = df_plot[col].apply(lambda x: x if x in top_categories else 'Other')
            logger.info(f"Column {col} has {df[col].nunique()} categories, showing top 15 only")
        else:
            df_plot = df
        
        plt.figure(figsize=(12, 6))
        
        # Basic count plot
        ax = sns.countplot(x=col, data=df_plot, order=df_plot[col].value_counts().index)
        
        # Rotate x labels if there are many categories
        if df_plot[col].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/categorical_{col}.png')
        plt.close()
        
        # Relationship with target
        if target_column in df.columns:
            # Plot percentage stacked bar chart
            plt.figure(figsize=(12, 6))
            
            # Calculate percentages
            ct = pd.crosstab(df_plot[col], df_plot[target_column], normalize='index') * 100
            
            # Plot
            ct.plot(kind='bar', stacked=True)
            
            # Rotate x labels if there are many categories
            if df_plot[col].nunique() > 5:
                plt.xticks(rotation=45, ha='right')
            
            plt.title(f'Percentage of {target_column} by {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel(f'Percentage of {target_column}', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/target_by_{col}.png')
            plt.close()
    
    logger.info(f"Categorical feature plots saved to {output_dir}")

def plot_correlation_matrix(df, target_column='Churn', output_dir='visualizations'):
    """
    Plot correlation matrix of numerical features
    
    Args:
        df: DataFrame with customer data
        target_column: Name of the target column
        output_dir: Directory to save visualizations
    """
    logger.info("Plotting correlation matrix")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Ensure target is included if it's numeric or boolean
    if target_column in df.columns:
        if df[target_column].dtype == 'bool':
            # Convert boolean to int for correlation
            df[target_column] = df[target_column].astype(int)
        
        if df[target_column].dtype in ('int64', 'float64'):
            if target_column not in numerical_cols:
                numerical_cols = numerical_cols.append(pd.Index([target_column]))
    
    # Skip if not enough numerical features
    if len(numerical_cols) < 2:
        logger.warning("Not enough numerical features for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
               linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()
    
    # Plot correlations with target if it exists
    if target_column in numerical_cols:
        # Sort features by correlation with target
        target_correlations = corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=target_correlations.values, y=target_correlations.index)
        plt.title(f'Correlation with {target_column}', fontsize=14)
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/target_correlations.png')
        plt.close()
    
    logger.info(f"Correlation plots saved to {output_dir}")

def main():
    """
    Main function to run data exploration
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore customer data for churn prediction')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--target', default='Churn', help='Name of the target column (default: Churn)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info("Starting data exploration")
    
    # Load the data
    df = load_data(args.input)
    
    # Generate data summary
    summary = data_summary(df, target_column=args.target)
    
    # Display summary
    print("\n=== Data Summary ===")
    print(f"Shape: {summary['shape'][0]} rows, {summary['shape'][1]} columns")
    print(f"Columns: {', '.join(summary['columns'])}")
    
    if args.target in df.columns:
        print(f"\nTarget ({args.target}) distribution:")
        for label, count in summary['target_distribution'].items():
            percentage = summary['target_percentage'][label]
            print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Identify potential issues
    issues = identify_potential_issues(df)
    
    # Display issues
    print("\n=== Potential Data Issues ===")
    
    if issues['missing_values']:
        print("\nMissing Values:")
        for col, pct in issues['missing_values'].items():
            print(f"  {col}: {pct:.2f}%")
    else:
        print("\nNo missing values found.")
    
    if issues['outliers']:
        print("\nOutliers detected in:")
        for col, stats in issues['outliers'].items():
            print(f"  {col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
    
    if issues['skewed_columns']:
        print("\nHighly skewed columns:")
        for col, skew in issues['skewed_columns'].items():
            direction = "right" if skew > 0 else "left"
            print(f"  {col}: {skew:.2f} ({direction}-skewed)")
    
    if issues['imbalanced_categories']:
        print("\nImbalanced categorical columns:")
        for col, stats in issues['imbalanced_categories'].items():
            print(f"  {col}: '{stats['dominant_category']}' accounts for {stats['percentage']:.1f}%")
    
    # Create visualizations
    plot_target_distribution(df, target_column=args.target, output_dir=args.output)
    plot_numerical_features(df, target_column=args.target, output_dir=args.output)
    plot_categorical_features(df, target_column=args.target, output_dir=args.output)
    plot_correlation_matrix(df, target_column=args.target, output_dir=args.output)
    
    # Save summary to file
    import json
    
    # Convert summary to JSON-serializable format
    for key in ['dtypes', 'numerical_stats']:
        if key in summary:
            summary[key] = {k: str(v) for k, v in summary[key].items()}
    
    with open(f"{args.output}/data_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data exploration completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
