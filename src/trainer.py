from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .model_selector import build_tabular_model
from .task_detector import TaskInfo
from .utils import TrainingHistory, get_logger


LOGGER = get_logger(__name__)


@dataclass
class TrainedModelResult:
    """Training output for one configuration."""

    config: Dict[str, Any]
    model_path: str
    history: TrainingHistory
    val_metric: float
    val_loss: float
    model_name: str


def _to_numpy(X: Any) -> np.ndarray:
    """Convert dense/sparse matrix to numpy array."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.array(X)


def _build_dataloader(X: Any, y: Any, batch_size: int, shuffle: bool = True) -> DataLoader:
    X_np = _to_numpy(X).astype(np.float32)
    y_np = np.array(y)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    if y_np.dtype.kind in {"i", "u"}:
        y_tensor = torch.tensor(y_np, dtype=torch.long)
    else:
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_models(
    task_info: TaskInfo,
    configs: List[Dict[str, Any]],
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    models_dir: str | Path,
    num_optuna_trials: int = 5,
    early_stopping_patience: int = 10,
    max_epochs: int = 100,
    progress_callback: Optional[Callable[[str, float, Dict[str, float]], None]] = None,
) -> List[TrainedModelResult]:
    """Train all configurations with Optuna fine-tuning and early stopping."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    all_results: List[TrainedModelResult] = []

    for cfg_idx, base_config in enumerate(configs, start=1):
        try:
            LOGGER.info("Training configuration %d/%d", cfg_idx, len(configs))

            def objective(trial: optuna.Trial) -> float:
                config = copy.deepcopy(base_config)
                config["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
                config["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
                config["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
                config["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)

                model = build_tabular_model(task_info, config)
                criterion: nn.Module
                is_regression = "regression" in task_info.task_type
                criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

                if config["optimizer"] == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)

                train_loader = _build_dataloader(train_data["X"], train_data["y"], config["batch_size"], True)
                val_loader = _build_dataloader(val_data["X"], val_data["y"], config["batch_size"], False)

                best_val = float("inf")
                best_metric = -float("inf")
                patience = 0
                history = TrainingHistory([], [], [], [])

                for epoch in range(min(max_epochs, config.get("epochs", 80))):
                    model.train()
                    tr_losses: List[float] = []
                    tr_true: List[float] = []
                    tr_pred: List[float] = []

                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        output = model(xb)
                        if is_regression:
                            loss = criterion(output.squeeze(-1), yb.float())
                            preds = output.squeeze(-1).detach().cpu().numpy()
                            tr_pred.extend(preds.tolist())
                            tr_true.extend(yb.cpu().numpy().tolist())
                        else:
                            loss = criterion(output, yb)
                            preds = torch.argmax(output, dim=1).detach().cpu().numpy()
                            tr_pred.extend(preds.tolist())
                            tr_true.extend(yb.cpu().numpy().tolist())

                        loss.backward()
                        optimizer.step()
                        tr_losses.append(float(loss.item()))

                    model.eval()
                    val_losses: List[float] = []
                    val_true: List[float] = []
                    val_pred: List[float] = []

                    with torch.no_grad():
                        for xb, yb in val_loader:
                            output = model(xb)
                            if is_regression:
                                loss = criterion(output.squeeze(-1), yb.float())
                                preds = output.squeeze(-1).cpu().numpy()
                            else:
                                loss = criterion(output, yb)
                                preds = torch.argmax(output, dim=1).cpu().numpy()

                            val_losses.append(float(loss.item()))
                            val_pred.extend(preds.tolist())
                            val_true.extend(yb.cpu().numpy().tolist())

                    train_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
                    val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

                    if is_regression:
                        train_metric = -float(mean_squared_error(tr_true, tr_pred))
                        val_metric = -float(mean_squared_error(val_true, val_pred))
                    else:
                        train_metric = float(accuracy_score(tr_true, tr_pred))
                        val_metric = float(accuracy_score(val_true, val_pred))

                    history.train_loss.append(train_loss)
                    history.val_loss.append(val_loss)
                    history.train_metric.append(train_metric)
                    history.val_metric.append(val_metric)

                    improved = val_loss < best_val if is_regression else val_metric > best_metric
                    if improved:
                        best_val = val_loss
                        best_metric = val_metric
                        patience = 0
                    else:
                        patience += 1

                    if progress_callback:
                        progress = (cfg_idx - 1 + (epoch + 1) / max(1, min(max_epochs, config.get("epochs", 80)))) / len(configs)
                        progress_callback(
                            f"config_{cfg_idx}",
                            progress,
                            {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss, "val_metric": val_metric},
                        )

                    if patience >= early_stopping_patience:
                        break

                trial.set_user_attr("history", history)
                trial.set_user_attr("best_val", best_val)
                trial.set_user_attr("best_metric", best_metric)
                return best_val if is_regression else (1.0 - best_metric)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=num_optuna_trials)

            best_params = study.best_trial.params
            tuned = {**base_config, **best_params}

            final_model = build_tabular_model(task_info, tuned)
            criterion = nn.MSELoss() if "regression" in task_info.task_type else nn.CrossEntropyLoss()
            optimizer = (
                torch.optim.Adam(final_model.parameters(), lr=tuned["learning_rate"])
                if tuned["optimizer"] == "Adam"
                else torch.optim.SGD(final_model.parameters(), lr=tuned["learning_rate"], momentum=0.9)
            )

            train_loader = _build_dataloader(train_data["X"], train_data["y"], tuned["batch_size"], True)
            val_loader = _build_dataloader(val_data["X"], val_data["y"], tuned["batch_size"], False)

            is_regression = "regression" in task_info.task_type
            best_metric = -float("inf")
            best_loss = float("inf")
            patience = 0
            best_state = copy.deepcopy(final_model.state_dict())
            history = TrainingHistory([], [], [], [])

            for epoch in range(min(max_epochs, tuned.get("epochs", 80))):
                final_model.train()
                tr_losses: List[float] = []
                tr_true: List[float] = []
                tr_pred: List[float] = []

                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    output = final_model(xb)
                    if is_regression:
                        loss = criterion(output.squeeze(-1), yb.float())
                        preds = output.squeeze(-1).detach().cpu().numpy()
                    else:
                        loss = criterion(output, yb)
                        preds = torch.argmax(output, dim=1).detach().cpu().numpy()

                    loss.backward()
                    optimizer.step()

                    tr_losses.append(float(loss.item()))
                    tr_pred.extend(preds.tolist())
                    tr_true.extend(yb.cpu().numpy().tolist())

                final_model.eval()
                val_losses: List[float] = []
                val_true: List[float] = []
                val_pred: List[float] = []

                with torch.no_grad():
                    for xb, yb in val_loader:
                        output = final_model(xb)
                        if is_regression:
                            loss = criterion(output.squeeze(-1), yb.float())
                            preds = output.squeeze(-1).cpu().numpy()
                        else:
                            loss = criterion(output, yb)
                            preds = torch.argmax(output, dim=1).cpu().numpy()

                        val_losses.append(float(loss.item()))
                        val_pred.extend(preds.tolist())
                        val_true.extend(yb.cpu().numpy().tolist())

                train_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                train_metric = -float(mean_squared_error(tr_true, tr_pred)) if is_regression else float(accuracy_score(tr_true, tr_pred))
                val_metric = -float(mean_squared_error(val_true, val_pred)) if is_regression else float(accuracy_score(val_true, val_pred))

                history.train_loss.append(train_loss)
                history.val_loss.append(val_loss)
                history.train_metric.append(train_metric)
                history.val_metric.append(val_metric)

                improved = val_loss < best_loss if is_regression else val_metric > best_metric
                if improved:
                    best_loss = val_loss
                    best_metric = val_metric
                    patience = 0
                    best_state = copy.deepcopy(final_model.state_dict())
                else:
                    patience += 1

                if patience >= early_stopping_patience:
                    break

            final_model.load_state_dict(best_state)

            timestamp = int(time.time())
            model_file = models_path / f"best_model_cfg{cfg_idx}_{timestamp}.pth"
            meta_file = models_path / f"best_model_cfg{cfg_idx}_{timestamp}.json"

            torch.save(final_model.state_dict(), model_file)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task_info": task_info.__dict__,
                        "config": tuned,
                        "feature_names": train_data.get("feature_names", []),
                    },
                    f,
                    indent=2,
                )

            # Save preprocessing artifacts for reproducibility
            if train_data.get("preprocessor") is not None:
                joblib.dump(train_data["preprocessor"], models_path / f"preprocessor_cfg{cfg_idx}_{timestamp}.joblib")
            if train_data.get("target_encoder") is not None:
                joblib.dump(train_data["target_encoder"], models_path / f"target_encoder_cfg{cfg_idx}_{timestamp}.joblib")

            all_results.append(
                TrainedModelResult(
                    config=tuned,
                    model_path=str(model_file),
                    history=history,
                    val_metric=float(best_metric),
                    val_loss=float(best_loss),
                    model_name="TabularANN",
                )
            )

        except RuntimeError as e:
            LOGGER.exception("RuntimeError while training config %d: %s", cfg_idx, e)
            if "out of memory" in str(e).lower():
                LOGGER.error("OOM detected for config %d. Skipping this configuration.", cfg_idx)
            continue
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Training failed for config %d: %s", cfg_idx, e)
            continue

    return all_results


def select_best_model(trained_models: List[TrainedModelResult], task_info: TaskInfo) -> TrainedModelResult:
    """Select best model using validation metric/loss."""
    if not trained_models:
        raise ValueError("No trained models available")

    if "regression" in task_info.task_type:
        return min(trained_models, key=lambda x: x.val_loss)
    return max(trained_models, key=lambda x: x.val_metric)


def evaluate_model(
    model: nn.Module,
    test_data: Dict[str, Any],
    task_info: TaskInfo,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Evaluate model and return metrics, confusion matrix, and predictions."""
    loader = _build_dataloader(test_data["X"], test_data["y"], batch_size=batch_size, shuffle=False)
    model.eval()

    y_true: List[float] = []
    y_pred: List[float] = []

    with torch.no_grad():
        for xb, yb in loader:
            output = model(xb)
            if "regression" in task_info.task_type:
                preds = output.squeeze(-1).cpu().numpy()
            else:
                preds = torch.argmax(output, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.cpu().numpy().tolist())

    if "regression" in task_info.task_type:
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "predictions": y_pred,
            "actual": y_true,
            "confusion_matrix": None,
        }

    labels = sorted(list(set(map(int, y_true)) | set(map(int, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm,
        "labels": labels,
        "predictions": y_pred,
        "actual": y_true,
    }
