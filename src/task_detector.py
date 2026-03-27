from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class TaskInfo:
    """Detected data and task information."""

    data_type: str
    task_type: str
    num_classes: int
    num_features: int
    target_column: Optional[str] = None


def detect_data_type(data: Any) -> str:
    """Detect whether input is tabular, image, or text."""
    if isinstance(data, pd.DataFrame):
        return "tabular"

    if isinstance(data, dict) and "images" in data:
        return "image"

    if isinstance(data, list) and data and isinstance(data[0], str):
        return "text"

    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return "tabular"
        if suffix in {".txt", ".jsonl"}:
            return "text"
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".zip"}:
            return "image"

    return "tabular"


def detect_task_type(data: Any, target: Optional[str]) -> str:
    """Infer the ML task type from data and target."""
    data_type = detect_data_type(data)

    if data_type == "image":
        if isinstance(data, dict) and data.get("bounding_boxes") is not None:
            return "object_detection"
        return "image_classification"

    if data_type == "text":
        sample = data[0].lower() if isinstance(data, list) and data else ""
        if any(w in sample for w in ["happy", "sad", "good", "bad", "love", "hate"]):
            return "sentiment_analysis"
        return "text_classification"

    if not isinstance(data, pd.DataFrame) or target is None or target not in data.columns:
        return "binary_classification"

    y = data[target]

    if pd.api.types.is_numeric_dtype(y):
        unique = y.nunique(dropna=True)
        if unique <= 2:
            return "binary_classification"
        if unique <= max(20, int(len(y) * 0.05)):
            return "multiclass_classification"
        return "regression"

    return "multiclass_classification" if y.nunique(dropna=True) > 2 else "binary_classification"


def get_task_info(data: Any, target: Optional[str] = None) -> TaskInfo:
    """Build a `TaskInfo` object from detected characteristics."""
    data_type = detect_data_type(data)
    task_type = detect_task_type(data, target)

    num_classes = 0
    num_features = 0

    if data_type == "tabular" and isinstance(data, pd.DataFrame):
        if target and target in data.columns:
            y = data[target]
            num_classes = int(y.nunique(dropna=True)) if "classification" in task_type else 1
            num_features = int(data.drop(columns=[target]).shape[1])
        else:
            num_features = int(data.shape[1])
    elif data_type == "image" and isinstance(data, dict):
        num_classes = len(set(data.get("labels", []))) if data.get("labels") else 0
        num_features = 224 * 224 * 3
    elif data_type == "text" and isinstance(data, list):
        num_features = int(np.mean([len(t.split()) for t in data])) if data else 0

    return TaskInfo(
        data_type=data_type,
        task_type=task_type,
        num_classes=max(num_classes, 1) if "classification" in task_type else num_classes,
        num_features=num_features,
        target_column=target,
    )


def task_info_to_dict(task_info: TaskInfo) -> Dict[str, Any]:
    """Serialize a `TaskInfo` object to dictionary."""
    return {
        "data_type": task_info.data_type,
        "task_type": task_info.task_type,
        "num_classes": task_info.num_classes,
        "num_features": task_info.num_features,
        "target_column": task_info.target_column,
    }
