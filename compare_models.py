#!/usr/bin/env python3
"""
Compare Old vs New Model Performance
====================================
This script runs the backtest on both the original and improved models,
then generates a comparison report.

Usage:
    python compare_models.py
"""

import os
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path

# === Fix Windows encoding issues ===
# Set UTF-8 encoding for stdout/stderr to handle emoji characters
if sys.platform == 'win32':
    try:
        # Python 3.7+
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configuration
FORECAST_DIR_OLD = "forecasts"
FORECAST_DIR_NEW = "forecasts_improved"
MODEL_NAME_OLD = "old_model"
MODEL_NAME_NEW = "new_model"
OUTPUT_DIR = "backtest_results"
VIDEO_DIR = "videos"

def run_backtest(forecast_dir, model_name):
    """Run the backtest script with specified parameters."""
    print(f"\n{'='*60}")
    print(f"Running backtest for {model_name}")
    print(f"Forecast directory: {forecast_dir}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python",
        "forecasting_backtest_Predictor_v2.py",
        "--forecast-dir", forecast_dir,
        "--model-name", model_name,
        "--output-dir", OUTPUT_DIR,
        "--video-dir", VIDEO_DIR
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: forecasting_backtest_Predictor_v2.py not found!")
        print("Make sure you're running from the project root directory.")
        return False

def load_metrics(model_name):
    """Load metrics JSON file for a model."""
    metrics_file = Path(OUTPUT_DIR) / f"metrics_{model_name}.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Metrics file not found: {metrics_file}")
        return None

def generate_comparison_report(old_metrics, new_metrics):
    """Generate a comparison report between two models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON REPORT")
    print("="*60)
    
    if old_metrics is None or new_metrics is None:
        print("Error: Could not load metrics for comparison.")
        return
    
    # Create comparison DataFrame
    comparison_data = {
        "Metric": [
            "Final Portfolio Value",
            "Total Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
            "Successful Buys",
            "Total Trades",
            "Beats SPY",
            "Return vs SPY (%)",
            "Number of Stocks",
            "Rejected Tickers"
        ],
        "Old Model": [
            f"{old_metrics['final_value']:.4f}",
            f"{old_metrics['total_return']*100:.2f}%",
            f"{old_metrics['sharpe_ratio']:.4f}",
            f"{old_metrics['max_drawdown']*100:.2f}%",
            f"{old_metrics['win_rate']*100:.2f}%",
            old_metrics['successful_buys'],
            old_metrics['total_buys'],
            "Yes" if old_metrics['beats_spy'] else "No",
            f"{old_metrics['return_over_spy']*100:.2f}%",
            old_metrics['num_stocks'],
            old_metrics['rejected_tickers']
        ],
        "New Model": [
            f"{new_metrics['final_value']:.4f}",
            f"{new_metrics['total_return']*100:.2f}%",
            f"{new_metrics['sharpe_ratio']:.4f}",
            f"{new_metrics['max_drawdown']*100:.2f}%",
            f"{new_metrics['win_rate']*100:.2f}%",
            new_metrics['successful_buys'],
            new_metrics['total_buys'],
            "Yes" if new_metrics['beats_spy'] else "No",
            f"{new_metrics['return_over_spy']*100:.2f}%",
            new_metrics['num_stocks'],
            new_metrics['rejected_tickers']
        ]
    }
    
    # Calculate improvements
    improvements = []
    for metric_key, display_name in [
        ('final_value', 'Final Portfolio Value'),
        ('total_return', 'Total Return'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('win_rate', 'Win Rate'),
    ]:
        old_val = old_metrics[metric_key]
        new_val = new_metrics[metric_key]
        
        if metric_key == 'max_drawdown':
            # For drawdown, less is better (closer to 0)
            improvement = (old_val - new_val) / abs(old_val) * 100 if old_val != 0 else 0
        else:
            # For other metrics, more is better
            improvement = (new_val - old_val) / abs(old_val) * 100 if old_val != 0 else 0
        
        improvements.append(f"{improvement:+.2f}%")
    
    comparison_data["Improvement"] = improvements + [""] * (len(comparison_data["Metric"]) - len(improvements))
    
    df = pd.DataFrame(comparison_data)
    
    # Print formatted table
    print("\n" + df.to_string(index=False))
    
    # Determine winner
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    old_return = old_metrics['total_return']
    new_return = new_metrics['total_return']
    old_sharpe = old_metrics['sharpe_ratio']
    new_sharpe = new_metrics['sharpe_ratio']
    
    if new_return > old_return:
        print(f"[WINNER] NEW MODEL WINS on Total Return")
        print(f"   Improvement: {(new_return - old_return)*100:.2f}%")
    elif old_return > new_return:
        print(f"[WINNER] OLD MODEL WINS on Total Return")
        print(f"   Difference: {(old_return - new_return)*100:.2f}%")
    else:
        print("[INFO] Models have identical Total Return")
    
    if new_sharpe > old_sharpe:
        print(f"[WINNER] NEW MODEL has better Sharpe Ratio")
        print(f"   Improvement: {new_sharpe - old_sharpe:.4f}")
    elif old_sharpe > new_sharpe:
        print(f"[WINNER] OLD MODEL has better Sharpe Ratio")
        print(f"   Difference: {old_sharpe - new_sharpe:.4f}")
    else:
        print("[INFO] Models have identical Sharpe Ratio")
    
    # Save comparison to CSV
    output_file = Path(OUTPUT_DIR) / "model_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n[SAVED] Comparison saved to: {output_file}")
    
    # Save detailed JSON comparison
    comparison_json = {
        "comparison_date": pd.Timestamp.now().isoformat(),
        "old_model": old_metrics,
        "new_model": new_metrics,
        "improvements": {
            "total_return_improvement_pct": (new_return - old_return) / abs(old_return) * 100 if old_return != 0 else 0,
            "sharpe_ratio_improvement": new_sharpe - old_sharpe,
            "final_value_improvement_pct": (new_metrics['final_value'] - old_metrics['final_value']) / old_metrics['final_value'] * 100 if old_metrics['final_value'] != 0 else 0,
        }
    }
    
    json_file = Path(OUTPUT_DIR) / "model_comparison.json"
    with open(json_file, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    print(f"[SAVED] Detailed comparison saved to: {json_file}")

def main():
    """Main function to run comparison."""
    print("="*60)
    print("MODEL COMPARISON TOOL")
    print("="*60)
    print(f"\nThis will run backtests on:")
    print(f"  1. Old Model: {FORECAST_DIR_OLD}")
    print(f"  2. New Model: {FORECAST_DIR_NEW}")
    print(f"\nResults will be saved to: {OUTPUT_DIR}")
    print(f"Videos will be saved to: {VIDEO_DIR}")
    
    # Check if forecast directories exist
    if not Path(FORECAST_DIR_OLD).exists():
        print(f"\n[ERROR] Forecast directory not found: {FORECAST_DIR_OLD}")
        print("Please run forecast.ipynb first to generate forecasts.")
        return
    
    if not Path(FORECAST_DIR_NEW).exists():
        print(f"\n[ERROR] Forecast directory not found: {FORECAST_DIR_NEW}")
        print("Please run forecast_improved.ipynb first to generate forecasts.")
        return
    
    # Run backtests
    print("\n" + "="*60)
    print("STEP 1: Running backtest on OLD model")
    print("="*60)
    success_old = run_backtest(FORECAST_DIR_OLD, MODEL_NAME_OLD)
    
    print("\n" + "="*60)
    print("STEP 2: Running backtest on NEW model")
    print("="*60)
    success_new = run_backtest(FORECAST_DIR_NEW, MODEL_NAME_NEW)
    
    if not success_old or not success_new:
        print("\n[ERROR] One or both backtests failed.")
        print("Please check the errors above and try again.")
        return
    
    # Load metrics
    print("\n" + "="*60)
    print("STEP 3: Generating comparison report")
    print("="*60)
    old_metrics = load_metrics(MODEL_NAME_OLD)
    new_metrics = load_metrics(MODEL_NAME_NEW)
    
    if old_metrics and new_metrics:
        generate_comparison_report(old_metrics, new_metrics)
        print("\n[SUCCESS] Comparison complete!")
        print(f"\nCheck the following files:")
        print(f"  - {OUTPUT_DIR}/metrics_{MODEL_NAME_OLD}.json")
        print(f"  - {OUTPUT_DIR}/metrics_{MODEL_NAME_NEW}.json")
        print(f"  - {OUTPUT_DIR}/model_comparison.csv")
        print(f"  - {OUTPUT_DIR}/model_comparison.json")
        print(f"  - {VIDEO_DIR}/random_vs_strategy_clean_{MODEL_NAME_OLD}.mp4")
        print(f"  - {VIDEO_DIR}/random_vs_strategy_clean_{MODEL_NAME_NEW}.mp4")
    else:
        print("\n[ERROR] Could not load metrics for comparison.")

if __name__ == "__main__":
    main()

