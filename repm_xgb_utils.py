"""
REPM Model Loading and Prediction Utilities

This module provides functions to load trained REPM models (XGBoost or Ridge)
and make predictions during simulation. It integrates with the existing UrbanSim framework.

Models:
- XGBoost: Used for segments with >= 100 samples
- Ridge: Used for segments with < 100 samples (better generalization for small samples)
"""

import os
import numpy as np
import pandas as pd
import yaml
import joblib
import xgboost as xgb
from pathlib import Path


class REPMXGBoostModel:
    """
    Wrapper class for REPM models (XGBoost or Ridge).

    This class handles loading trained models and making predictions
    for real estate price estimation. Automatically detects model type
    from metadata and loads the appropriate model.

    Model Types:
    - XGBoost: For segments with >= 100 samples
    - Ridge: For segments with < 100 samples
    """

    def __init__(self, model_name, model_dir="./configs/repm_xgb/"):
        """
        Initialize the REPM XGBoost model.

        Parameters:
        model_name (str): Name of the model (e.g., 'res_repm381')
        model_dir (str): Directory containing the trained models
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None
        self.metadata = None
        self._load_model()

    def _load_model(self):
        """Load the trained model (XGBoost or Ridge) and metadata."""
        model_path = os.path.join(self.model_dir, self.model_name)

        # Load metadata first to determine model type
        metadata_path = os.path.join(model_path, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = joblib.load(metadata_path)
        self.feature_names = self.metadata['feature_names']
        model_type = self.metadata.get('model_type', 'xgboost')

        # Load model based on type
        if model_type == 'xgboost':
            xgb_model_path = os.path.join(model_path, "xgb_model.json")
            if not os.path.exists(xgb_model_path):
                raise FileNotFoundError(f"XGBoost model file not found: {xgb_model_path}")

            self.model = xgb.XGBRegressor()
            self.model.load_model(xgb_model_path)
        elif model_type == 'ridge':
            sklearn_model_path = os.path.join(model_path, "sklearn_model.pkl")
            if not os.path.exists(sklearn_model_path):
                raise FileNotFoundError(f"Ridge model file not found: {sklearn_model_path}")

            self.model = joblib.load(sklearn_model_path)
        elif model_type == 'dummy':
            sklearn_model_path = os.path.join(model_path, "sklearn_model.pkl")
            if not os.path.exists(sklearn_model_path):
                raise FileNotFoundError(f"Dummy model file not found: {sklearn_model_path}")

            self.model = joblib.load(sklearn_model_path)
            print(f"Loaded {model_type} model {self.model_name} (mean predictor fallback)")
        else:
            # Fallback for any sklearn model
            sklearn_model_path = os.path.join(model_path, "sklearn_model.pkl")
            if not os.path.exists(sklearn_model_path):
                raise FileNotFoundError(f"Sklearn model file not found: {sklearn_model_path}")

            self.model = joblib.load(sklearn_model_path)

        if model_type != 'dummy':
            print(f"Loaded {model_type} model {self.model_name} with {len(self.feature_names)} features")
        print(f"Model R²: {self.metadata['metrics']['r2_val']:.4f}")

    def predict(self, X):
        """
        Make predictions using the trained XGBoost model.

        Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix

        Returns:
        np.ndarray: Predicted values (log-transformed price)
        """
        # Ensure X is a DataFrame with correct feature order
        if isinstance(X, pd.DataFrame):
            # Select only the features used by the model
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                print(f"Warning: Missing {len(missing_features)} features, filling with 0")
                for feat in missing_features:
                    X[feat] = 0

            X_subset = X[self.feature_names].values
        else:
            X_subset = X

        # Make prediction
        predictions = self.model.predict(X_subset)

        return predictions

    def get_feature_importance(self, top_n=20):
        """
        Get top N most important features.

        Parameters:
        top_n (int): Number of top features to return

        Returns:
        pd.DataFrame: DataFrame with feature names and importance scores
        """
        importances = self.metadata['feature_importance']
        # Filter out zero importance (dummy models)
        importances = {k: v for k, v in importances.items() if v > 0}
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(sorted_features[:top_n], columns=['feature', 'importance'])
        if len(df) > 0 and df['importance'].sum() > 0:
            df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        else:
            df['importance_pct'] = 0.0

        return df


def load_repm_xgb_model(model_name, model_dir="./configs/repm_xgb/"):
    """
    Load a trained REPM XGBoost model.

    Parameters:
    model_name (str): Name of the model (e.g., 'res_repm381')
    model_dir (str): Directory containing the trained models

    Returns:
    REPMXGBoostModel: Loaded model wrapper
    """
    return REPMXGBoostModel(model_name, model_dir)


def predict_repm_xgb(cfg, tbl, nodes, out_fname, model_dir="./configs/repm_xgb/"):
    """
    Simulate hedonic prices using trained XGBoost models.

    This function is designed to integrate with the existing UrbanSim simulation pipeline.
    It replaces the utils.hedonic_simulate function for XGBoost models.

    Parameters:
    cfg (str): Path to the YAML config file
    tbl (orca.DataFrameWrapper): Buildings table
    nodes (orca.DataFrameWrapper): Nodes table (e.g., nodes_walk)
    out_fname (str): Output column name for predictions
    model_dir (str): Directory containing the trained models

    Returns:
    None: Updates the buildings table in place
    """
    import orca
    from urbansim.utils import misc
    from utils import to_frame, deal_with_nas

    # Load config
    cfg_full = misc.config(cfg)
    model_name = os.path.basename(cfg).replace('.yaml', '')

    # Load model
    model = load_repm_xgb_model(model_name, model_dir)

    # Get data
    df = to_frame([tbl, nodes], cfg_full)
    df = deal_with_nas(df)

    # Make predictions
    predictions_log = model.predict(df)

    # Apply inverse transform (expm1 is inverse of log1p)
    predictions = np.expm1(predictions_log)

    # Clip predictions to reasonable range
    predictions = np.clip(predictions, 1, 700)

    # Update table
    tbl.update_col_from_series(out_fname, pd.Series(predictions, index=df.index), cast=True)

    print(f"Updated {out_fname} for {len(predictions)} buildings using {model_name}")


def list_available_models(model_dir="./configs/repm_xgb/"):
    """
    List all available trained XGBoost REPM models.

    Parameters:
    model_dir (str): Directory containing the trained models

    Returns:
    dict: Dictionary with model names and their metadata
    """
    models = {}

    if not os.path.exists(model_dir):
        return models

    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            metadata_path = os.path.join(item_path, "metadata.pkl")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                models[item] = {
                    'n_features': len(metadata['feature_names']),
                    'r2_val': metadata['metrics']['r2_val'],
                    'sample_size': metadata['metrics']['sample_size'],
                }

    return models


def compare_repm_models(model_dir="./configs/repm_xgb/"):
    """
    Print a comparison of all trained REPM models.

    Parameters:
    model_dir (str): Directory containing the trained models
    """
    models = list_available_models(model_dir)

    if not models:
        print("No trained models found in", model_dir)
        return

    print(f"\n{'='*80}")
    print(f"REPM XGBoost Models Summary ({len(models)} models)")
    print(f"{'='*80}")
    print(f"{'Model Name':<25} {'Features':<10} {'Samples':<12} {'R² Val':<10}")
    print(f"{'-'*80}")

    for model_name, info in sorted(models.items()):
        print(f"{model_name:<25} {info['n_features']:<10} {info['sample_size']:<12} {info['r2_val']:.4f}")

    print(f"{'-'*80}")
    avg_r2 = np.mean([m['r2_val'] for m in models.values()])
    print(f"Average R²: {avg_r2:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example usage and testing
    print("REPM XGBoost Utility Module")
    print("="*60)

    # List available models
    compare_repm_models()

    # Example: Load and use a specific model
    # model = load_repm_xgb_model('res_repm381')
    # importance = model.get_feature_importance(top_n=10)
    # print(importance)
