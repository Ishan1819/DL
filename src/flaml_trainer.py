import os
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from flaml import AutoML

def train_flaml_baseline(
    task_info: Any,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    models_dir: Path,
    time_budget: int = 60
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a FLAML AutoML model as a baseline for tabular datasets.
    """
    if task_info.data_type != "tabular":
        return None, {}

    # Identify task type
    is_classification = "classification" in task_info.task_type
    task = "classification" if is_classification else "regression"
    
    # Prepare data
    X_train = train_data["X"]
    y_train = train_data["y"]
    X_val = val_data["X"]
    y_val = val_data["y"]

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    automl = AutoML()
    
    # AutoML settings
    settings = {
        "time_budget": time_budget,
        "metric": "accuracy" if is_classification else "rmse",
        "task": task,
        "eval_method": "holdout",
        "X_val": X_val,
        "y_val": y_val,
        "log_file_name": str(models_dir / "flaml.log"),
    }

    # Fit model
    automl.fit(X_train=X_train, y_train=y_train, **settings)

    # Evaluate on val
    y_pred = automl.predict(X_val)
    if is_classification:
        val_metric = np.mean(y_pred == y_val)
    else:
        val_metric = np.sqrt(np.mean((y_pred - y_val)**2))

    metrics = {
        "val_metric": float(val_metric),
        "best_estimator": str(automl.best_estimator),
        "best_config": automl.best_config,
        "time_budget": time_budget
    }

    # Save model
    model_path = models_dir / "flaml_baseline.joblib"
    joblib.dump(automl, model_path)

    return automl, metrics
