from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler

from .utils import get_logger


LOGGER = get_logger(__name__)


def extract_meta_features(df: pd.DataFrame, target: str) -> Dict[str, float]:
    """Extract normalized meta-features for tabular data using PyMFE."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    X = df.drop(columns=[target])
    y = df[target]

    X_numeric = X.copy()
    for col in X_numeric.columns:
        if not pd.api.types.is_numeric_dtype(X_numeric[col]):
            X_numeric[col] = pd.factorize(X_numeric[col].astype(str))[0]

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y.astype(str))[0]

    mfe = MFE(groups=["statistical", "info-theory"])  # mean, std, skewness, kurtosis, entropy
    mfe.fit(X_numeric.to_numpy(), np.array(y))
    names, values = mfe.extract()

    features = {k: float(np.nanmean(v)) if isinstance(v, (list, tuple, np.ndarray)) else float(v) for k, v in zip(names, values)}

    features.update(
        {
            "num_features": float(X.shape[1]),
            "num_samples": float(X.shape[0]),
            "num_classes": float(pd.Series(y).nunique()),
            "class_balance": float(pd.Series(y).value_counts(normalize=True).max()),
        }
    )

    clean_keys = []
    clean_values = []
    for key, value in features.items():
        if np.isfinite(value):
            clean_keys.append(key)
            clean_values.append(value)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(clean_values).reshape(-1, 1)).reshape(-1)

    normalized = {k: float(v) for k, v in zip(clean_keys, scaled)}
    LOGGER.info("Extracted %d normalized meta-features", len(normalized))
    return normalized
