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

# Lazy import to avoid circular dependency at module level
def _get_suggest_config_fn():
    try:
        from .llm_generator import suggest_config_with_llm  # pylint: disable=import-outside-toplevel
        return suggest_config_with_llm
    except Exception:  # pylint: disable=broad-except
        return None


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


def _infer_input_dim(X: Any) -> int:
    """Infer feature dimension from processed design matrix."""
    if hasattr(X, "shape") and len(getattr(X, "shape")) >= 2:
        return int(X.shape[1])
    arr = np.array(X)
    if arr.ndim == 1:
        return 1
    return int(arr.shape[1])


def _hidden_profile_candidates(base_hidden: List[int] | Any) -> Dict[str, List[int]]:
    """Create BO-searchable hidden-layer profiles around a warm-start architecture."""
    base = [max(16, int(h)) for h in (list(base_hidden) if base_hidden else [128, 64, 32])]
    small = [max(16, int(h * 0.75)) for h in base]
    large = [min(1024, int(h * 1.25)) for h in base]
    shallow = base[:-1] if len(base) > 2 else base
    deep = base + [max(16, base[-1] // 2)] if len(base) < 5 else base

    candidates = {
        "small": small,
        "base": base,
        "large": large,
        "shallow": shallow,
        "deep": deep,
    }

    deduped: Dict[str, List[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for name, dims in candidates.items():
        key = tuple(dims)
        if key not in seen:
            seen.add(key)
            deduped[name] = dims
    return deduped


def _deepen_hidden_dims(hidden_dims: List[int] | Any) -> List[int]:
    """Add one extra hidden layer for adaptive retry when accuracy drops."""
    dims = [max(16, int(h)) for h in (list(hidden_dims) if hidden_dims else [128, 64, 32])]
    if len(dims) >= 6:
        return dims
    tail = dims[-1]
    extra = max(16, tail // 2)
    if extra >= tail:
        extra = max(16, tail - 16)
    return dims + [extra]


def _train_final_pass(
    task_info: TaskInfo,
    config: Dict[str, Any],
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    max_epochs: int,
    early_stopping_patience: int,
) -> tuple[nn.Module, TrainingHistory, float, float]:
    """Train one final model pass for a concrete config with early stopping."""
    final_model = build_tabular_model(task_info, config)
    criterion = nn.MSELoss() if "regression" in task_info.task_type else nn.CrossEntropyLoss()
    optimizer = (
        torch.optim.Adam(final_model.parameters(), lr=config["learning_rate"])
        if config["optimizer"] == "Adam"
        else torch.optim.SGD(final_model.parameters(), lr=config["learning_rate"], momentum=0.9)
    )

    train_loader = _build_dataloader(train_data["X"], train_data["y"], config["batch_size"], True)
    val_loader = _build_dataloader(val_data["X"], val_data["y"], config["batch_size"], False)

    is_regression = "regression" in task_info.task_type
    best_metric = -float("inf")
    best_loss = float("inf")
    patience = 0
    best_state = copy.deepcopy(final_model.state_dict())
    history = TrainingHistory([], [], [], [])

    for _ in range(min(max_epochs, int(config.get("epochs", 80)))):
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
    return final_model, history, float(best_metric), float(best_loss)


def _train_tabular_tpe(
    task_info: TaskInfo,
    warm_start_config: Dict[str, Any],
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    models_dir: Path,
    n_trials: int = 5,
    early_stopping_patience: int = 10,
    max_epochs: int = 100,
    progress_callback: Optional[Callable[[str, float, Dict[str, float]], None]] = None,
) -> List[TrainedModelResult]:
    """
    Single Optuna TPE study warm-started from LCBench surrogate recommendation.

    Flow
    ----
    Trial 0  : LCBench surrogate config (enqueued — not random).
    Trials 1+: TPE sampler suggests based on previous trial results.
    Each trial trains for eval_epochs (fast comparison).
    Finally: best config is retrained fully (max_epochs + early stopping).

    This is the standard SMBO warm-start pattern from Auto-sklearn 2.0.
    """
    is_regression = "regression" in task_info.task_type
    input_dim = _infer_input_dim(train_data["X"])
    eval_epochs = max(5, max_epochs // 5)          # short eval: ~20% of full budget
    hidden_profiles = _hidden_profile_candidates(warm_start_config.get("hidden_dims", [128, 64, 32]))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=1)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # ── Enqueue LCBench surrogate as trial 0 ──────────────────────────────
    study.enqueue_trial({
        "learning_rate": float(warm_start_config["learning_rate"]),
        "batch_size":    int(warm_start_config["batch_size"]),
        "dropout":       float(warm_start_config["dropout"]),
        "optimizer":     warm_start_config.get("optimizer", "Adam"),
    })

    # ── Objective (short training for fast comparison) ────────────────────
    def objective(trial: optuna.Trial) -> float:
        lr       = trial.suggest_float("learning_rate", 1e-5, 0.1, log=True)
        bs       = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        dropout  = trial.suggest_float("dropout", 0.0, 0.7)
        opt_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

        hidden_profile = trial.suggest_categorical("hidden_profile", list(hidden_profiles.keys()))
        config = {
            **warm_start_config,
            "learning_rate": lr,
            "batch_size": bs,
            "dropout": dropout,
            "optimizer": opt_name,
            "hidden_dims": hidden_profiles[hidden_profile],
            "hidden_profile": hidden_profile,
            "input_dim": input_dim,
            "epochs": eval_epochs,
        }

        model     = build_tabular_model(task_info, config)
        criterion: nn.Module = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=lr)
            if opt_name == "Adam"
            else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        )
        train_loader = _build_dataloader(train_data["X"], train_data["y"], bs, True)
        val_loader   = _build_dataloader(val_data["X"], val_data["y"], bs, False)

        best_val_loss   = float("inf")
        best_val_metric = -float("inf")
        patience        = 0
        history         = TrainingHistory([], [], [], [])

        for epoch in range(eval_epochs):
            model.train()
            tr_losses: List[float] = []
            tr_true:   List[float] = []
            tr_pred:   List[float] = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                if is_regression:
                    loss = criterion(out.squeeze(-1), yb.float())
                    tr_pred.extend(out.squeeze(-1).detach().cpu().numpy().tolist())
                else:
                    loss = criterion(out, yb)
                    tr_pred.extend(torch.argmax(out, dim=1).detach().cpu().numpy().tolist())
                tr_true.extend(yb.cpu().numpy().tolist())
                loss.backward()
                optimizer.step()
                tr_losses.append(float(loss.item()))

            model.eval()
            val_losses: List[float] = []
            val_true:   List[float] = []
            val_pred:   List[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    out = model(xb)
                    if is_regression:
                        v_loss = criterion(out.squeeze(-1), yb.float())
                        val_pred.extend(out.squeeze(-1).cpu().numpy().tolist())
                    else:
                        v_loss = criterion(out, yb)
                        val_pred.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())
                    val_true.extend(yb.cpu().numpy().tolist())
                    val_losses.append(float(v_loss.item()))

            train_loss   = float(np.mean(tr_losses)) if tr_losses else 0.0
            val_loss_ep  = float(np.mean(val_losses)) if val_losses else float("inf")
            train_metric = (
                -float(mean_squared_error(tr_true, tr_pred)) if is_regression
                else float(accuracy_score(tr_true, tr_pred))
            )
            val_metric = (
                -float(mean_squared_error(val_true, val_pred)) if is_regression
                else float(accuracy_score(val_true, val_pred))
            )
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss_ep)
            history.train_metric.append(train_metric)
            history.val_metric.append(val_metric)

            improved = val_loss_ep < best_val_loss if is_regression else val_metric > best_val_metric
            if improved:
                best_val_loss, best_val_metric, patience = val_loss_ep, val_metric, 0
            else:
                patience += 1

            if progress_callback:
                frac = (trial.number * eval_epochs + epoch + 1) / (n_trials * eval_epochs)
                progress_callback(
                    f"tpe_trial_{trial.number + 1}",
                    min(frac * 0.8, 0.8),
                    {"epoch": float(epoch + 1), "train_loss": train_loss,
                     "val_loss": val_loss_ep, "val_metric": val_metric},
                )

            # Report intermediate values for pruning
            score = best_val_loss if is_regression else (1.0 - best_val_metric)
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if patience >= early_stopping_patience:
                break

        trial.set_user_attr("val_metric", best_val_metric)
        trial.set_user_attr("val_loss",   best_val_loss)
        trial.set_user_attr("history",    history)
        trial.set_user_attr("config",     config)
        return best_val_loss if is_regression else (1.0 - best_val_metric)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    # ── Collect short-training results for sidebar comparison ─────────────
    trial_results: List[TrainedModelResult] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_results.append(TrainedModelResult(
            config={**t.user_attrs.get("config", warm_start_config), "config_id": t.number + 1},
            model_path="",   # short-trained — no persistent file
            history=t.user_attrs.get("history", TrainingHistory([], [], [], [])),
            val_metric=float(t.user_attrs.get("val_metric", -1.0)),
            val_loss=float(t.user_attrs.get("val_loss", float("inf"))),
            model_name="TabularANN",
        ))

    # ── Full retrain of the best TPE config ───────────────────────────────
    best_trial  = study.best_trial
    best_config = {**best_trial.user_attrs.get("config", warm_start_config), "epochs": max_epochs}
    best_config.setdefault("input_dim", input_dim)
    LOGGER.info(
        "[TPE] Full retrain — lr=%.5f bs=%d dropout=%.3f opt=%s hidden=%s",
        best_config["learning_rate"], best_config["batch_size"],
        best_config["dropout"], best_config.get("optimizer", "Adam"),
        best_config.get("hidden_dims"),
    )

    final_model = build_tabular_model(task_info, best_config)
    criterion   = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    opt_obj     = (
        torch.optim.Adam(final_model.parameters(), lr=best_config["learning_rate"])
        if best_config.get("optimizer", "Adam") == "Adam"
        else torch.optim.SGD(final_model.parameters(), lr=best_config["learning_rate"], momentum=0.9)
    )
    train_loader = _build_dataloader(train_data["X"], train_data["y"], best_config["batch_size"], True)
    val_loader   = _build_dataloader(val_data["X"], val_data["y"], best_config["batch_size"], False)

    best_metric  = -float("inf")
    best_loss    = float("inf")
    patience     = 0
    best_state   = copy.deepcopy(final_model.state_dict())
    full_history = TrainingHistory([], [], [], [])

    for epoch in range(max_epochs):
        final_model.train()
        tr_losses2: List[float] = []
        tr_true2:   List[float] = []
        tr_pred2:   List[float] = []
        for xb, yb in train_loader:
            opt_obj.zero_grad()
            out = final_model(xb)
            if is_regression:
                loss = criterion(out.squeeze(-1), yb.float())
                tr_pred2.extend(out.squeeze(-1).detach().cpu().numpy().tolist())
            else:
                loss = criterion(out, yb)
                tr_pred2.extend(torch.argmax(out, dim=1).detach().cpu().numpy().tolist())
            tr_true2.extend(yb.cpu().numpy().tolist())
            loss.backward()
            opt_obj.step()
            tr_losses2.append(float(loss.item()))

        final_model.eval()
        val_losses2: List[float] = []
        val_true2:   List[float] = []
        val_pred2:   List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = final_model(xb)
                if is_regression:
                    v_loss = criterion(out.squeeze(-1), yb.float())
                    val_pred2.extend(out.squeeze(-1).cpu().numpy().tolist())
                else:
                    v_loss = criterion(out, yb)
                    val_pred2.extend(torch.argmax(out, dim=1).cpu().numpy().tolist())
                val_true2.extend(yb.cpu().numpy().tolist())
                val_losses2.append(float(v_loss.item()))

        train_loss_f  = float(np.mean(tr_losses2)) if tr_losses2 else 0.0
        val_loss_f    = float(np.mean(val_losses2)) if val_losses2 else float("inf")
        train_metric_f = (
            -float(mean_squared_error(tr_true2, tr_pred2)) if is_regression
            else float(accuracy_score(tr_true2, tr_pred2))
        )
        val_metric_f = (
            -float(mean_squared_error(val_true2, val_pred2)) if is_regression
            else float(accuracy_score(val_true2, val_pred2))
        )
        full_history.train_loss.append(train_loss_f)
        full_history.val_loss.append(val_loss_f)
        full_history.train_metric.append(train_metric_f)
        full_history.val_metric.append(val_metric_f)

        improved = val_loss_f < best_loss if is_regression else val_metric_f > best_metric
        if improved:
            best_loss, best_metric, patience = val_loss_f, val_metric_f, 0
            best_state = copy.deepcopy(final_model.state_dict())
        else:
            patience += 1

        if progress_callback:
            frac = 0.8 + (epoch + 1) / max(1, max_epochs) * 0.2
            progress_callback(
                "full_retrain",
                min(frac, 1.0),
                {"epoch": float(epoch + 1), "train_loss": train_loss_f,
                 "val_loss": val_loss_f, "val_metric": val_metric_f},
            )

        if patience >= early_stopping_patience:
            break

    final_model.load_state_dict(best_state)
    timestamp  = int(time.time())
    model_file = models_dir / f"best_model_tpe_{timestamp}.pth"
    meta_file  = models_dir / f"best_model_tpe_{timestamp}.json"
    torch.save(final_model.state_dict(), model_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_info": task_info.__dict__,
                "config": best_config,
                "feature_names": train_data.get("feature_names", []),
                "optuna_n_trials": n_trials,
                "warm_start": "lcbench_surrogate",
            },
            f, indent=2,
        )
    if train_data.get("preprocessor") is not None:
        joblib.dump(train_data["preprocessor"], models_dir / f"preprocessor_tpe_{timestamp}.joblib")
    if train_data.get("target_encoder") is not None:
        joblib.dump(train_data["target_encoder"], models_dir / f"target_encoder_tpe_{timestamp}.joblib")

    # Append fully-trained result — will have best val_metric → selected by select_best_model()
    trial_results.append(TrainedModelResult(
        config={**best_config, "config_id": len(trial_results) + 1},
        model_path=str(model_file),
        history=full_history,
        val_metric=float(best_metric),
        val_loss=float(best_loss),
        model_name="TabularANN [TPE best]",
    ))
    LOGGER.info("[TPE] Done. Best val_metric=%.4f after full retrain.", best_metric)
    return trial_results


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
    warm_start_config: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[str] = None,
    llm_api_keys: Optional[Dict[str, str]] = None,
    meta_features: Optional[Dict[str, float]] = None,
) -> List[TrainedModelResult]:
    """Train configurations with Optuna fine-tuning and early stopping.

    For tabular data with a warm_start_config (from LCBench surrogate),
    delegates to _train_tabular_tpe() which runs a single unified TPE study
    warm-started from the surrogate recommendation.  For non-tabular data,
    falls back to the original per-config Optuna loop.
    """
    # Keep single-study warm-start TPE path only when a single config is requested.
    if task_info.data_type == "tabular" and warm_start_config is not None and len(configs) <= 1:
        return _train_tabular_tpe(
            task_info=task_info,
            warm_start_config=warm_start_config,
            train_data=train_data,
            val_data=val_data,
            models_dir=Path(models_dir),
            n_trials=num_optuna_trials,
            early_stopping_patience=early_stopping_patience,
            max_epochs=max_epochs,
            progress_callback=progress_callback,
        )
    # ── Original per-config loop (non-tabular / no warm-start) ──────────
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    input_dim = _infer_input_dim(train_data["X"])

    all_results: List[TrainedModelResult] = []
    previous_val_metric: Optional[float] = None
    # LLM warm-start: suggestion from prior results to enqueue as trial 0
    llm_enqueue_params: Optional[Dict[str, Any]] = None
    _suggest_fn = _get_suggest_config_fn() if (llm_provider and llm_api_keys) else None

    for cfg_idx, base_config in enumerate(configs, start=1):
        try:
            LOGGER.info("Training configuration %d/%d", cfg_idx, len(configs))
            hidden_profiles = _hidden_profile_candidates(base_config.get("hidden_dims", [128, 64, 32]))

            def objective(trial: optuna.Trial) -> float:
                config = copy.deepcopy(base_config)
                config["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
                config["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
                config["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
                config["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
                hidden_profile = trial.suggest_categorical("hidden_profile", list(hidden_profiles.keys()))
                config["hidden_dims"] = hidden_profiles[hidden_profile]
                config["hidden_profile"] = hidden_profile
                config["input_dim"] = input_dim

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
                objective_value = best_val if is_regression else (1.0 - best_metric)
                LOGGER.info(
                    "[CFG %d][TRIAL %d] objective=%.4f val_metric=%.4f params=%s",
                    cfg_idx,
                    trial.number,
                    objective_value,
                    best_metric,
                    {
                        "learning_rate": config["learning_rate"],
                        "batch_size": config["batch_size"],
                        "optimizer": config["optimizer"],
                        "dropout": round(float(config["dropout"]), 4),
                        "hidden_profile": config.get("hidden_profile", "base"),
                    },
                )
                return objective_value

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")

            # ── LLM warm-start: enqueue LLM-suggested params as trial 0 ────
            if llm_enqueue_params is not None:
                valid_keys = {"learning_rate", "batch_size", "optimizer", "dropout", "hidden_profile"}
                safe_params = {k: v for k, v in llm_enqueue_params.items() if k in valid_keys}
                # Map hidden_dims → closest hidden_profile name
                if "hidden_dims" in llm_enqueue_params and "hidden_profile" not in safe_params:
                    llm_hdims = list(llm_enqueue_params["hidden_dims"])
                    best_profile = min(
                        hidden_profiles.keys(),
                        key=lambda name: sum(
                            abs(a - b) for a, b in zip(hidden_profiles[name], llm_hdims)
                        )
                    )
                    safe_params["hidden_profile"] = best_profile
                # Clamp values to valid ranges
                if "learning_rate" in safe_params:
                    safe_params["learning_rate"] = max(1e-4, min(1e-2, float(safe_params["learning_rate"])))
                if "batch_size" in safe_params:
                    safe_params["batch_size"] = int(safe_params["batch_size"])
                    if safe_params["batch_size"] not in [32, 64, 128]:
                        safe_params["batch_size"] = min([32, 64, 128], key=lambda x: abs(x - safe_params["batch_size"]))
                if "dropout" in safe_params:
                    safe_params["dropout"] = max(0.1, min(0.5, float(safe_params["dropout"])))
                if safe_params:
                    try:
                        study.enqueue_trial(safe_params)
                        LOGGER.info("[CFG %d] LLM warm-start enqueued: %s", cfg_idx, safe_params)
                    except Exception as eq_err:  # pylint: disable=broad-except
                        LOGGER.warning("[CFG %d] Could not enqueue LLM suggestion: %s", cfg_idx, eq_err)

            study.optimize(objective, n_trials=num_optuna_trials)

            best_trial = study.best_trial
            best_obj = float(best_trial.value)
            best_metric_for_log = (
                -1.0 * best_obj
                if "regression" in task_info.task_type
                else (1.0 - best_obj)
            )
            LOGGER.info(
                "[CFG %d] best_trial=%d objective=%.4f mapped_val_metric=%.4f params=%s",
                cfg_idx,
                best_trial.number,
                best_obj,
                best_metric_for_log,
                best_trial.params,
            )

            best_params = best_trial.params
            tuned = {**base_config, **best_params}
            tuned["input_dim"] = input_dim
            final_model, history, best_metric, best_loss = _train_final_pass(
                task_info=task_info,
                config=tuned,
                train_data=train_data,
                val_data=val_data,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
            )

            # Adaptive retry: if current classification accuracy drops vs previous config,
            # deepen the network and extend epochs, then retrain once.
            is_regression = "regression" in task_info.task_type
            if (not is_regression) and previous_val_metric is not None and best_metric < previous_val_metric:
                adaptive_tuned = copy.deepcopy(tuned)
                adaptive_tuned["hidden_dims"] = _deepen_hidden_dims(adaptive_tuned.get("hidden_dims", [128, 64, 32]))
                base_epochs = int(adaptive_tuned.get("epochs", 80))
                adaptive_tuned["epochs"] = min(max_epochs, max(base_epochs + 2, int(base_epochs * 1.5)))

                LOGGER.info(
                    "[ADAPT] Config %d val_acc %.4f below previous %.4f. Retrying with hidden=%s epochs=%d",
                    cfg_idx,
                    best_metric,
                    previous_val_metric,
                    adaptive_tuned["hidden_dims"],
                    adaptive_tuned["epochs"],
                )

                adaptive_model, adaptive_history, adaptive_metric, adaptive_loss = _train_final_pass(
                    task_info=task_info,
                    config=adaptive_tuned,
                    train_data=train_data,
                    val_data=val_data,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                )

                if adaptive_metric > best_metric:
                    final_model = adaptive_model
                    history = adaptive_history
                    best_metric = adaptive_metric
                    best_loss = adaptive_loss
                    tuned = adaptive_tuned
                    tuned["adaptive_retry"] = True
                else:
                    tuned["adaptive_retry"] = False
            elif not is_regression:
                tuned["adaptive_retry"] = False

            tuned["best_trial_objective"] = best_obj
            tuned["mapped_val_accuracy"] = best_metric_for_log if not is_regression else None
            tuned["mapped_val_metric"] = best_metric_for_log
            tuned["final_retrain_val_accuracy"] = best_metric if not is_regression else None
            tuned["final_retrain_val_metric"] = best_metric

            if not is_regression:
                previous_val_metric = best_metric

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

            # \u2500\u2500 LLM suggestion for next config's warm-start \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            if _suggest_fn is not None and cfg_idx < len(configs):
                try:
                    prior = [
                        {"config": r.config, "val_metric": r.val_metric}
                        for r in all_results
                    ]
                    llm_enqueue_params = _suggest_fn(
                        previous_results=prior,
                        meta_features=meta_features,
                        llm_provider=llm_provider,
                        api_keys=llm_api_keys,
                    )
                    LOGGER.info(
                        "[CFG %d] LLM suggested next config: %s",
                        cfg_idx,
                        {k: v for k, v in llm_enqueue_params.items() if k != "reason"},
                    )
                except Exception as llm_err:  # pylint: disable=broad-except
                    LOGGER.warning("[CFG %d] LLM suggestion failed: %s", cfg_idx, llm_err)
                    llm_enqueue_params = None


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

    # Prefer models with persisted checkpoints. Some search-only trials can have empty paths.
    persisted_models = [m for m in trained_models if m.model_path and Path(m.model_path).exists()]
    candidates = persisted_models if persisted_models else trained_models

    if "regression" in task_info.task_type:
        return min(candidates, key=lambda x: x.val_loss)
    return max(candidates, key=lambda x: x.val_metric)


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
