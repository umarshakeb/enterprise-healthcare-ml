import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_FILE = OUTPUT_DIR / "feature_baseline.json"

def generate_feature_baseline(X_train, numeric_features, n_bins=10):
    """
    Generate training-data feature distributions and bin edges
    for PSI-based drift monitoring.

    The generated JSON contains one baseline entry per numerical
    feature.
    """
    baseline = {}
    for feature in numeric_features:
        if feature not in X_train.columns:
            raise ValueError(
                f"Feature : {feature} not present in X_train"
            )
        values = pd.to_numeric(X_train[feature], errors='coerce').dropna()

        if values.empty:
            print(f"Skipping {feature}: no valid numeric values found")
            continue

        unique_values = np.sort(values.unique())
        if len(unique_values)<2:
            print(f"Skipping {feature}: feweer than 2 unique values")
            continue

        actual_bins = min(n_bins, len(unique_values)-1)
        quantiles = np.linspace(0,1,actual_bins+1)
        bin_edges = np.quantile(values, quantiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            print(f"Skipping {feature} : unable to create valid bins")
            continue

        counts, _ = np.histogram(values,bins=bin_edges)
        distribution = counts/counts.sum()
        baseline[feature] = {
            "distribution" : distribution.tolist(),
            "bin_edges" : bin_edges.tolist()
        }
    
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline,f, indent=2)
    
    print(f"Feature baseline generated : {BASELINE_FILE}")
    return str(BASELINE_FILE)