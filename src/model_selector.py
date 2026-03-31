from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .task_detector import TaskInfo
from .surrogate_selector import recommend_ann_config


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


class TabularResNetBlock(nn.Module):
    """Residual block for TabularResNet."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid BatchNorm 1D error when batch size is 1 during evaluation/SHAP
        if x.size(0) > 1:
            out = self.bn1(self.lin1(x))
            out = self.relu(out)
            out = self.dropout(out)
            out = self.bn2(self.lin2(out))
            return self.relu(x + out)
        else:
            out = self.lin1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.lin2(out)
            return self.relu(x + out)


class TabularResNet(nn.Module):
    """Residual network for tabular datasets."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        regression: bool = False,
    ) -> None:
        super().__init__()
        width = hidden_dims[0] if hidden_dims else 128
        num_blocks = max(2, len(hidden_dims))
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            TabularResNetBlock(width, dropout) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(width, output_dim)
        self.regression = regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        return self.output_layer(out)


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
        # ------------------------------------------------------------------
        # Use LCBench surrogate (k-NN warm-start + RF scoring) to derive
        # architecture and hyperparameters.  Falls back to formula-based
        # sizing if the surrogate pkl files have not been generated yet.
        # ------------------------------------------------------------------
        _mf: Dict[str, float] = meta_features or {}
        # Ensure basic structural meta-features are always present
        _mf.setdefault("num_features", float(task_info.num_features))
        _mf.setdefault("num_classes", float(task_info.num_classes))

        ann_cfg = recommend_ann_config(_mf)
        hidden_dims: List[int] = ann_cfg["hidden_dims"]
        dropout: float = ann_cfg["dropout"]
        source_note = ann_cfg.get("source", "lcbench_surrogate")

        if "regression" in task_info.task_type:
            recs.append(
                ModelRecommendation(
                    model_name="TabularANNRegressor",
                    architecture={
                        "layers": [task_info.num_features] + hidden_dims + [1],
                        "activation": "ReLU",
                        "dropout": dropout,
                        "output": "Linear",
                    },
                    loss_name="MSELoss",
                    notes=f"Standard PyTorch ANN for tabular regression [{source_note}]",
                )
            )
            recs.append(
                ModelRecommendation(
                    model_name="TabularResNetRegressor",
                    architecture={
                        "blocks": max(2, len(hidden_dims)),
                        "width": hidden_dims[0] if hidden_dims else 128,
                        "dropout": dropout,
                        "output": "Linear",
                    },
                    loss_name="MSELoss",
                    notes=f"Residual Network (TabularResNet) for regression [{source_note}]",
                )
            )
        else:
            recs.append(
                ModelRecommendation(
                    model_name="TabularANNClassifier",
                    architecture={
                        "layers": [task_info.num_features] + hidden_dims + [task_info.num_classes],
                        "activation": "ReLU",
                        "dropout": dropout,
                        "output": "Softmax",
                    },
                    loss_name="CrossEntropyLoss",
                    notes=f"Standard PyTorch ANN for tabular classification [{source_note}]",
                )
            )
            recs.append(
                ModelRecommendation(
                    model_name="TabularResNetClassifier",
                    architecture={
                        "blocks": max(2, len(hidden_dims)),
                        "width": hidden_dims[0] if hidden_dims else 128,
                        "dropout": dropout,
                        "output": "Softmax",
                    },
                    loss_name="CrossEntropyLoss",
                    notes=f"Residual Network (TabularResNet) for classification [{source_note}]",
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


def generate_configs(task_info: TaskInfo, meta_features: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    """Generate 3-5 candidate hyperparameter configurations.

    For tabular data, the base config is derived from the LCBench surrogate
    (or formula fallback) seeded by meta_features.  Neighbouring configs are
    created by perturbing LR, batch_size, and dropout slightly so Optuna /
    the trainer still has a search space around the surrogate recommendation.
    """
    optimizers = ["Adam", "SGD"]
    epochs = 80 if task_info.data_type == "tabular" else 15
    configs: List[Dict[str, Any]] = []

    if task_info.data_type == "tabular":
        _mf: Dict[str, float] = meta_features or {}
        _mf.setdefault("num_features", float(task_info.num_features))
        _mf.setdefault("num_classes", float(task_info.num_classes))

        ann_cfg = recommend_ann_config(_mf)
        base_lr: float = ann_cfg["learning_rate"]
        base_bs: int = ann_cfg["batch_size"]
        base_hidden: List[int] = ann_cfg["hidden_dims"]
        base_dropout: float = ann_cfg["dropout"]

        # LR perturbations: surrogate recommendation + one order lower/higher
        lr_variants = [base_lr, base_lr * 0.1, base_lr * 10.0, base_lr * 0.3, base_lr * 3.0]
        bs_variants = [base_bs, max(16, base_bs // 2), min(512, base_bs * 2), base_bs, max(16, base_bs // 2)]
        dropout_variants = [base_dropout, base_dropout, max(0.0, base_dropout - 0.1),
                             min(0.7, base_dropout + 0.1), base_dropout]

        for i in range(5):
            configs.append(
                {
                    "config_id": i + 1,
                    "architecture": "ANN" if i % 2 == 0 else "ResNet",
                    "learning_rate": round(lr_variants[i], 6),
                    "batch_size": int(bs_variants[i]),
                    "optimizer": optimizers[i % len(optimizers)],
                    "epochs": epochs,
                    "dropout": round(dropout_variants[i], 3),
                    "hidden_dims": base_hidden,
                }
            )
    else:
        # Non-tabular: keep original cycling defaults (CNN/RNN/Transformer)
        learning_rates = [1e-3, 1e-2, 1e-4]
        batch_sizes = [32, 64, 128]
        for i in range(5):
            configs.append(
                {
                    "config_id": i + 1,
                    "learning_rate": learning_rates[i % len(learning_rates)],
                    "batch_size": batch_sizes[i % len(batch_sizes)],
                    "optimizer": optimizers[i % len(optimizers)],
                    "epochs": min(20, epochs + i),
                    "dropout": 0.2 + 0.05 * (i % 3),
                    "hidden_dims": [128, 64, 32],
                }
            )
    return configs


def build_tabular_model(task_info: TaskInfo, config: Dict[str, Any]) -> nn.Module:
    """Create tabular ANN or ResNet model from detected task and config."""
    regression = "regression" in task_info.task_type
    output_dim = 1 if regression else max(task_info.num_classes, 2)
    input_dim = int(config.get("input_dim", task_info.num_features))
    hidden_dims = config.get("hidden_dims", [128, 64, 32])
    dropout = float(config.get("dropout", 0.3))
    
    arch_type = config.get("architecture", "ANN")
    
    if arch_type == "ResNet":
        return TabularResNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            regression=regression,
        )
    else:
        return TabularANN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            regression=regression,
        )
