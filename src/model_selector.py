from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .task_detector import TaskInfo


@dataclass
class ModelRecommendation:
    """Container for model recommendations."""

    model_name: str
    architecture: Dict[str, Any]
    loss_name: str
    notes: str


class TabularANN(nn.Module):
    """Feed-forward network for tabular classification/regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        regression: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)
        self.regression = regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMTimeSeries(nn.Module):
    """Simple LSTM/BiLSTM head for sequence prediction/classification."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        output_dim: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def recommend_model(task_info: TaskInfo, meta_features: Dict[str, float] | None = None) -> List[ModelRecommendation]:
    """Recommend model architectures by data and task type."""
    recs: List[ModelRecommendation] = []

    if task_info.data_type == "tabular":
        if "regression" in task_info.task_type:
            recs.append(
                ModelRecommendation(
                    model_name="TabularANNRegressor",
                    architecture={
                        "layers": [task_info.num_features, 64, 32, 1],
                        "activation": "ReLU",
                        "dropout": 0.2,
                        "output": "Linear",
                    },
                    loss_name="MSELoss",
                    notes="PyTorch ANN for tabular regression",
                )
            )
        else:
            recs.append(
                ModelRecommendation(
                    model_name="TabularANNClassifier",
                    architecture={
                        "layers": [task_info.num_features, 128, 64, 32, task_info.num_classes],
                        "activation": "ReLU",
                        "dropout": 0.3,
                        "output": "Softmax",
                    },
                    loss_name="CrossEntropyLoss",
                    notes="PyTorch ANN for tabular classification",
                )
            )

    elif task_info.data_type == "image":
        if task_info.task_type == "object_detection":
            recs.append(
                ModelRecommendation(
                    model_name="YOLOv8",
                    architecture={"variant": "yolov8n.pt", "pretrained": True},
                    loss_name="YOLO internal",
                    notes="Ultralytics detector",
                )
            )
        else:
            recs.extend(
                [
                    ModelRecommendation(
                        model_name="ResNet50",
                        architecture={"backbone": "resnet50", "pretrained": True},
                        loss_name="CrossEntropyLoss",
                        notes="Torchvision transfer learning",
                    ),
                    ModelRecommendation(
                        model_name="EfficientNetB0",
                        architecture={"backbone": "efficientnet_b0", "pretrained": True},
                        loss_name="CrossEntropyLoss",
                        notes="Torchvision transfer learning",
                    ),
                ]
            )

    elif task_info.data_type == "text":
        if task_info.task_type == "sentiment_analysis":
            recs.append(
                ModelRecommendation(
                    model_name="RoBERTa",
                    architecture={"checkpoint": "roberta-base"},
                    loss_name="CrossEntropyLoss",
                    notes="HuggingFace transformer",
                )
            )
        elif task_info.task_type == "time_series":
            recs.extend(
                [
                    ModelRecommendation(
                        model_name="LSTM",
                        architecture={"hidden_dim": 128, "num_layers": 1, "bidirectional": False},
                        loss_name="CrossEntropyLoss",
                        notes="Custom PyTorch LSTM",
                    ),
                    ModelRecommendation(
                        model_name="BiLSTM",
                        architecture={"hidden_dim": 128, "num_layers": 1, "bidirectional": True},
                        loss_name="CrossEntropyLoss",
                        notes="Custom PyTorch Bi-LSTM",
                    ),
                ]
            )
        else:
            recs.extend(
                [
                    ModelRecommendation(
                        model_name="DistilBERT",
                        architecture={"checkpoint": "distilbert-base-uncased"},
                        loss_name="CrossEntropyLoss",
                        notes="HuggingFace transformer",
                    ),
                    ModelRecommendation(
                        model_name="BERT-base",
                        architecture={"checkpoint": "bert-base-uncased"},
                        loss_name="CrossEntropyLoss",
                        notes="HuggingFace transformer",
                    ),
                ]
            )

    return recs


def generate_configs(task_info: TaskInfo) -> List[Dict[str, Any]]:
    """Generate 3-5 candidate hyperparameter configurations."""
    learning_rates = [1e-3, 1e-2, 1e-4]
    batch_sizes = [32, 64, 128]
    optimizers = ["Adam", "SGD"]

    epochs = 80 if task_info.data_type == "tabular" else 15

    configs: List[Dict[str, Any]] = []
    for i in range(5):
        configs.append(
            {
                "config_id": i + 1,
                "learning_rate": learning_rates[i % len(learning_rates)],
                "batch_size": batch_sizes[i % len(batch_sizes)],
                "optimizer": optimizers[i % len(optimizers)],
                "epochs": epochs if task_info.data_type == "tabular" else min(20, epochs + i),
                "dropout": 0.2 + 0.05 * (i % 3),
                "hidden_dims": [128, 64, 32] if "classification" in task_info.task_type else [64, 32],
            }
        )
    return configs


def build_tabular_model(task_info: TaskInfo, config: Dict[str, Any]) -> nn.Module:
    """Create tabular ANN model from detected task and config."""
    regression = "regression" in task_info.task_type
    output_dim = 1 if regression else max(task_info.num_classes, 2)
    return TabularANN(
        input_dim=task_info.num_features,
        output_dim=output_dim,
        hidden_dims=config.get("hidden_dims", [128, 64, 32]),
        dropout=float(config.get("dropout", 0.3)),
        regression=regression,
    )
