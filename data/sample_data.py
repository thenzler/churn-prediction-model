#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate sample data for churn prediction model.
This script creates synthetic customer data that can be used for testing and demonstration.
"""

import pandas as pd
import numpy as np
import os
import argparse

def generate_sample_data(num_samples=1000, seed=42):
    """
    Generate synthetic customer data for churn prediction
    
    Args:
        num_samples: Number of customer records to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic customer data
    """
    np.random.seed(seed)
    
    # Customer IDs
    customer_ids = [f'CUST_{i:06d}' for i in range(1, num_samples + 1)]
    
    # Demographic features
    ages = np.random.normal(42, 15, num_samples).astype(int)
    ages = np.clip(ages, 18, 90)  # Clip to reasonable age range
    
    genders = np.random.choice(['Male', 'Female', 'Other'], num_samples, p=[0.48, 0.48, 0.04])
    
    regions = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central'], 
        num_samples, 
        p=[0.2, 0.25, 0.2, 0.25, 0.1]
    )
    
    # Contract features
    contract_types = np.random.choice(
        ['Monthly', 'Annual', 'Two-year'], 
        num_samples, 
        p=[0.6, 0.3, 0.1]
    )
    
    # Tenure in months (with distribution skewed towards new customers)
    tenures = np.random.exponential(scale=20, size=num_samples).astype(int)
    tenures = np.clip(tenures, 1, 120)  # Clip to 1-120 months
    
    # Usage metrics (based on tenure and contract)
    monthly_charges = np.zeros(num_samples)
    
    # Set base charges by contract type
    for i, contract in enumerate(contract_types):
        if contract == 'Monthly':
            monthly_charges[i] = np.random.uniform(20, 50)
        elif contract == 'Annual':
            monthly_charges[i] = np.random.uniform(40, 80)
        else:  # Two-year
            monthly_charges[i] = np.random.uniform(60, 100)
    
    # Add some random variation
    monthly_charges += np.random.normal(0, 5, num_samples)
    monthly_charges = np.clip(monthly_charges, 15, 120)
    
    # Total charges based on tenure and monthly charges
    total_charges = monthly_charges * tenures
    
    # Service features
    has_phone = np.random.choice([0, 1], num_samples, p=[0.3, 0.7])
    has_internet = np.random.choice([0, 1], num_samples, p=[0.2, 0.8])
    has_tv = np.random.choice([0, 1], num_samples, p=[0.5, 0.5])
    has_streaming = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    
    # Total number of services
    total_services = has_phone + has_internet + has_tv + has_streaming
    
    # Interaction features
    support_calls = np.random.poisson(lam=1.5, size=num_samples)
    support_calls = np.clip(support_calls, 0, 10)
    
    support_calls_per_month = support_calls / np.clip(tenures, 1, None)
    
    # Late payments
    late_payments = np.random.poisson(lam=0.3, size=num_samples)
    late_payments = np.clip(late_payments, 0, 5)
    
    # Determine churn probability based on features
    # Higher probability for customers with:
    # - Monthly contracts
    # - High support calls per month
    # - Low total services
    # - Late payments
    # - High monthly charges
    
    churn_prob = 0.1  # Base probability
    
    # Adjust by contract type
    contract_factor = np.zeros(num_samples)
    contract_factor[contract_types == 'Monthly'] = 0.15
    contract_factor[contract_types == 'Annual'] = 0.05
    contract_factor[contract_types == 'Two-year'] = 0.02
    
    # Adjust by support calls
    support_factor = 0.05 * support_calls_per_month
    
    # Adjust by services
    services_factor = 0.05 * (4 - total_services)
    
    # Adjust by late payments
    late_factor = 0.08 * late_payments
    
    # Adjust by tenure (new customers are more likely to churn)
    tenure_factor = np.zeros(num_samples)
    tenure_factor[tenures <= 6] = 0.1
    tenure_factor[(tenures > 6) & (tenures <= 12)] = 0.05
    tenure_factor[tenures > 12] = 0.02
    
    # Calculate final churn probability
    churn_prob = churn_prob + contract_factor + support_factor + services_factor + late_factor + tenure_factor
    churn_prob = np.clip(churn_prob, 0.01, 0.9)  # Clip to reasonable range
    
    # Generate churn based on probability
    churn = np.random.random(num_samples) < churn_prob
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'region': regions,
        'contract_type': contract_types,
        'tenure_months': tenures,
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'has_phone_service': has_phone,
        'has_internet_service': has_internet,
        'has_tv_service': has_tv,
        'has_streaming_service': has_streaming,
        'total_services': total_services,
        'support_calls': support_calls,
        'support_calls_per_month': support_calls_per_month.round(4),
        'late_payments': late_payments,
        'churn_probability': churn_prob.round(4),
        'Churn': churn.astype(int)
    })
    
    return df

def main():
    """
    Main function to generate and save sample data
    """
    parser = argparse.ArgumentParser(description='Generate sample data for churn prediction')
    parser.add_argument('--output', default='customer_data.csv', help='Output file path (default: customer_data.csv)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save data
    df = generate_sample_data(args.samples, args.seed)
    df.to_csv(args.output, index=False)
    
    print(f"Generated {len(df)} synthetic customer records")
    print(f"Churn rate: {df['Churn'].mean() * 100:.2f}%")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
