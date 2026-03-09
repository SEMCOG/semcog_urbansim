#!/usr/bin/env python
"""
REPM Model Analysis: Compare XGBoost vs Lasso Performance

This script compares the performance of XGBoost REPM models against the old
Lasso regression models, generating detailed statistics and per-model comparisons.

Usage:
    python notebooks/repm_analysis.py
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import os
import glob


def load_xgb_results_from_log(log_file):
    """Load XGBoost results from training log file."""
    xgb_results = {}

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse the training table
    in_table = False
    for line in lines:
        if '[3/4] Training models...' in line:
            in_table = True
            continue
        if in_table and line.strip().startswith('-'):
            continue
        if in_table and 'Stage 4' in line or '[4/4]' in line:
            break
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 5:
                model_name = parts[0]
                try:
                    r2 = float(parts[3])
                    xgb_results[model_name] = r2
                except ValueError:
                    pass

    return xgb_results


def load_lasso_results(config_dir):
    """Load Lasso model results from YAML config files."""
    lasso_results = {}
    config_path = Path(config_dir)

    for yaml_file in config_path.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            if 'fit_rsquared' in data and data['fit_rsquared'] is not None:
                lasso_results[yaml_file.stem] = float(data['fit_rsquared'])
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}")

    return lasso_results


def compare_models(xgb_results, lasso_results):
    """Compare XGBoost and Lasso results per model."""
    comparisons = []

    for model in sorted(xgb_results.keys()):
        xgb_r2 = xgb_results[model]
        lasso_r2 = lasso_results.get(model, None)

        if lasso_r2 is not None:
            delta = xgb_r2 - lasso_r2

            comparison = {
                'model': model,
                'xgb_r2': xgb_r2,
                'lasso_r2': lasso_r2,
                'delta': delta,
                'is_residential': model.startswith('res_'),
                'improved': delta > 0.01,
                'worse': delta < -0.01,
                'same': abs(delta) <= 0.01,
            }
            comparisons.append(comparison)

    return pd.DataFrame(comparisons)


def print_summary_stats(df):
    """Print summary statistics for XGBoost and Lasso models."""
    print("="*80)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*80)

    res_df = df[df['is_residential']]
    nonres_df = df[~df['is_residential']]

    print("\nResidential Models:")
    print(f"  XGBoost:  R² = {res_df['xgb_r2'].mean():.4f} ± {res_df['xgb_r2'].std():.4f}  "
          f"[{res_df['xgb_r2'].min():.4f}, {res_df['xgb_r2'].max():.4f}]")
    print(f"  Lasso:    R² = {res_df['lasso_r2'].mean():.4f} ± {res_df['lasso_r2'].std():.4f}  "
          f"[{res_df['lasso_r2'].min():.4f}, {res_df['lasso_r2'].max():.4f}]")
    print(f"  Improvement: +{res_df['delta'].mean():.4f}")

    print("\nNon-Residential Models:")
    print(f"  XGBoost:  R² = {nonres_df['xgb_r2'].mean():.4f} ± {nonres_df['xgb_r2'].std():.4f}  "
          f"[{nonres_df['xgb_r2'].min():.4f}, {nonres_df['xgb_r2'].max():.4f}]")
    print(f"  Lasso:    R² = {nonres_df['lasso_r2'].mean():.4f} ± {nonres_df['lasso_r2'].std():.4f}  "
          f"[{nonres_df['lasso_r2'].min():.4f}, {nonres_df['lasso_r2'].max():.4f}]")
    print(f"  Improvement: +{nonres_df['delta'].mean():.4f}")

    print("\nAll Models:")
    print(f"  XGBoost:  R² = {df['xgb_r2'].mean():.4f} ± {df['xgb_r2'].std():.4f}")
    print(f"  Lasso:    R² = {df['lasso_r2'].mean():.4f} ± {df['lasso_r2'].std():.4f}")
    print(f"  Improvement: +{df['delta'].mean():.4f} (+{df['delta'].mean() / df['lasso_r2'].mean() * 100:.1f}%)")

    n_improved = df['improved'].sum()
    n_worse = df['worse'].sum()
    n_same = df['same'].sum()

    print(f"\nModel Changes:")
    print(f"  Improved:  {n_improved}/{len(df)} ({n_improved/len(df)*100:.1f}%)")
    print(f"  Worse:     {n_worse}/{len(df)} ({n_worse/len(df)*100:.1f}%)")
    print(f"  Same:      {n_same}/{len(df)} ({n_same/len(df)*100:.1f}%)")


def print_per_model_comparison(df, top_n=10):
    """Print detailed per-model comparison."""
    print("\n" + "="*100)
    print(f"PER-MODEL COMPARISON (Top {top_n} Improvements and Declines)")
    print("="*100)

    # Sort by delta
    df_sorted = df.sort_values('delta', ascending=False)

    print(f"\n{'Model':<20} {'XGBoost':<10} {'Lasso':<10} {'Delta':<12} {'Type'}")
    print("-"*100)

    for _, row in df_sorted.iterrows():
        marker = ">>>" if abs(row['delta']) > 0.3 else ""
        status = "✓" if row['improved'] else ("✗" if row['worse'] else "=")
        model_type = "Res" if row['is_residential'] else "NonRes"
        print(f"{row['model']:<20} {row['xgb_r2']:<10.4f} {row['lasso_r2']:<10.4f} "
              f"{row['delta']:+<12.4f} {status} {model_type} {marker}")

    # Top improvements
    print(f"\nTop {top_n} Improvements:")
    print("-"*100)
    top_improved = df_sorted.nlargest(top_n, 'delta')
    for _, row in top_improved.iterrows():
        model_type = "Residential" if row['is_residential'] else "Non-Residential"
        print(f"  {row['model']:<20} +{row['delta']:.4f}  ({model_type})")

    # Worst declines
    print(f"\nWorst {top_n} Declines:")
    print("-"*100)
    worst = df_sorted.nsmallest(top_n, 'delta')
    for _, row in worst.iterrows():
        model_type = "Residential" if row['is_residential'] else "Non-Residential"
        print(f"  {row['model']:<20} {row['delta']:.4f}  ({model_type})")


def export_comparison_csv(df, output_file):
    """Export comparison results to CSV."""
    df_export = df.sort_values('delta', ascending=False)
    df_export.to_csv(output_file, index=False)
    print(f"\nComparison exported to: {output_file}")


def main():
    """Main analysis function."""
    # Paths
    log_files = glob.glob("/mnt/semcog_urbansim/runs/training_logs/repm_train_*.txt")
    log_file = max(log_files, key=os.path.getmtime)  # Use latest log file
    print(f"Using log file: {log_file}")
    lasso_config_dir = "/mnt/semcog_urbansim/configs/repm_2050/"
    output_csv = "/mnt/semcog_urbansim/notebooks/repm_comparison.csv"

    print("REPM Model Analysis: XGBoost vs Lasso")
    print("="*80)

    # Load results
    print("\nLoading XGBoost results from log...")
    xgb_results = load_xgb_results_from_log(log_file)
    print(f"  Loaded {len(xgb_results)} XGBoost models")

    print("\nLoading Lasso results from configs...")
    lasso_results = load_lasso_results(lasso_config_dir)
    print(f"  Loaded {len(lasso_results)} Lasso models")

    # Compare
    print("\nComparing models...")
    df = compare_models(xgb_results, lasso_results)
    print(f"  Matched {len(df)} models")

    # Print summaries
    print_summary_stats(df)
    print_per_model_comparison(df, top_n=10)

    # Export
    export_comparison_csv(df, output_csv)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
