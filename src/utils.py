from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


@dataclass
class TrainingHistory:
    """Container for training curves."""

    train_loss: List[float]
    val_loss: List[float]
    train_metric: List[float]
    val_metric: List[float]


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger for the project."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger by name."""
    return logging.getLogger(name)


def load_config(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_training_curves(
    history: TrainingHistory,
    output_path: str | os.PathLike[str],
    metric_name: str = "accuracy",
) -> str:
    """Save training and validation curves."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.train_loss, label="Train Loss")
    axes[0].plot(history.val_loss, label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.train_metric, label=f"Train {metric_name.title()}")
    axes[1].plot(history.val_metric, label=f"Val {metric_name.title()}")
    axes[1].set_title(f"{metric_name.title()} Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name.title())
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return str(output)


def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: str | os.PathLike[str],
) -> str:
    """Save confusion matrix heatmap to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return str(output)


def save_text_report_pdf(report_text: str, output_path: str | os.PathLike[str]) -> str:
    """Render a plain text report into a basic PDF file via matplotlib."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    wrapped = report_text.splitlines()
    if not wrapped:
        wrapped = ["No report content available."]

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.clf()
    y = 0.98
    for line in wrapped:
        fig.text(0.05, y, line[:140], fontsize=9, va="top")
        y -= 0.02
        if y < 0.03:
            break
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return str(output)
