#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the complete churn prediction pipeline.
This script executes all steps from data generation to model training and evaluation.
"""

import os
import argparse
import logging
import subprocess
import time
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command):
    """
    Run a shell command and log its output
    
    Args:
        command: Command to run
        
    Returns:
        Command's return code
    """
    logger.info(f"Running command: {command}")
    
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(process.stdout)
        
        if process.stderr:
            logger.warning(process.stderr)
        
        return process.returncode
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)
        return e.returncode

def main():
    """
    Run the complete churn prediction pipeline
    """
    parser = argparse.ArgumentParser(description='Run the churn prediction pipeline')
    parser.add_argument('--data-dir', default='data', help='Directory for data files')
    parser.add_argument('--model-dir', default='model', help='Directory for model files')
    parser.add_argument('--viz-dir', default='visualizations', help='Directory for visualizations')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--skip-data-generation', action='store_true', help='Skip data generation step')
    parser.add_argument('--skip-exploration', action='store_true', help='Skip data exploration step')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # File paths
    data_file = os.path.join(args.data_dir, 'customer_data.csv')
    
    logger.info("Starting churn prediction pipeline")
    start_time = time.time()
    
    # Step 1: Generate sample data
    if not args.skip_data_generation:
        logger.info("Step 1: Generating sample data")
        cmd = f"python data/sample_data.py --output {data_file} --samples {args.samples}"
        if run_command(cmd) != 0:
            logger.error("Data generation failed. Exiting pipeline.")
            return 1
    else:
        logger.info("Skipping data generation step as requested")
    
    # Step 2: Explore data
    if not args.skip_exploration:
        logger.info("Step 2: Exploring data")
        cmd = f"python explore_data.py --input {data_file} --output {args.viz_dir}"
        if run_command(cmd) != 0:
            logger.warning("Data exploration failed, but continuing with pipeline.")
    else:
        logger.info("Skipping data exploration step as requested")
    
    # Step 3: Train model
    logger.info("Step 3: Training model")
    cmd = f"python train_model.py --input {data_file} --output {args.model_dir}"
    if args.optimize:
        cmd += " --optimize"
    
    if run_command(cmd) != 0:
        logger.error("Model training failed. Exiting pipeline.")
        return 1
    
    # Step 4: Make predictions on test data
    logger.info("Step 4: Testing predictions")
    model_file = os.path.join(args.model_dir, 'churn_model_latest.pkl')
    predictions_file = os.path.join(args.data_dir, 'predictions.csv')
    
    cmd = f"python predict_churn.py --model {model_file} --input {data_file} --output {predictions_file} --explain"
    if run_command(cmd) != 0:
        logger.error("Prediction testing failed. Exiting pipeline.")
        return 1
    
    # Done
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
    return 0

if __name__ == "__main__":
    sys.exit(main())
