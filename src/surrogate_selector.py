"""
surrogate_selector.py
=====================
Runtime module: loads the pre-trained LCBench surrogate and meta-store,
then recommends ANN hyperparameters for a new tabular dataset using:

  1. k-NN warm-start  – cosine similarity over LCBench meta-features
                        → top-k best configs as candidates
  2. Surrogate scoring – RF predicts val_accuracy for each candidate
                        → pick the highest-scoring config

Falls back to formula-based sizing if pkl artefacts are not found.
"""
from __future__ import annotations

import functools
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
_META_PKL = ROOT / "data" / "lcbench_meta.pkl"
_SURROGATE_PKL = ROOT / "data" / "lcbench_surrogate.pkl"

_META_FEATURE_KEYS = [
    "num_samples",
    "num_features",
    "num_classes",
    "imbalance_ratio",
    "missing_ratio",
]

_CONFIG_KEYS = [
    "num_layers",
    "max_units",
    "learning_rate",
    "dropout",
    "batch_size",
]


# ---------------------------------------------------------------------------
# Lazy loaders (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_artifacts() -> Optional[Tuple[Dict, Any]]:
    """Return (meta_store, surrogate) or None if pkl files are missing."""
    if not _META_PKL.exists() or not _SURROGATE_PKL.exists():
        return None
    with open(_META_PKL, "rb") as f:
        meta_store: Dict = pickle.load(f)
    with open(_SURROGATE_PKL, "rb") as f:
        surrogate = pickle.load(f)
    return meta_store, surrogate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_meta_vec(meta_features: Dict[str, float]) -> np.ndarray:
    """Project a meta-features dict to the fixed-length meta vector."""
    return np.array(
        [meta_features.get(k, 0.0) for k in _META_FEATURE_KEYS], dtype=float
    )


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _config_to_hidden_dims(num_layers: int, max_units: int) -> List[int]:
    """Convert LCBench num_layers/max_units to a list of hidden dims (halving)."""
    layers = []
    units = int(max_units)
    for i in range(int(num_layers)):
        layers.append(max(8, units // (2 ** i)))
    return layers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def knn_candidates(
    meta_features: Dict[str, float],
    k: int = 3,
) -> List[Dict[str, float]]:
    """
    Find top-k most similar LCBench datasets by cosine similarity on
    meta-features and return their best configs.
    """
    result = _load_artifacts()
    if result is None:
        return []
    meta_store, _ = result
    query_vec = _to_meta_vec(meta_features)

    scored: List[Tuple[float, Dict[str, float]]] = []
    for ds_name, ds_data in meta_store.items():
        ds_vec = _to_meta_vec(ds_data["meta_features"])
        sim = _cosine_sim(query_vec, ds_vec)
        scored.append((sim, ds_data["best_config"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [cfg for _, cfg in scored[:k]]


def score_configs(
    candidates: List[Dict[str, float]],
    meta_features: Dict[str, float],
) -> List[Tuple[float, Dict[str, float]]]:
    """
    Use the surrogate RF to predict val_accuracy for each candidate config.
    Returns list of (predicted_accuracy, config) sorted best-first.
    """
    result = _load_artifacts()
    if result is None or not candidates:
        return []
    _, surrogate = result
    meta_vec = _to_meta_vec(meta_features).tolist()

    rows = []
    for cfg in candidates:
        cfg_vec = [float(cfg.get(k, 0.0)) for k in _CONFIG_KEYS]
        rows.append(meta_vec + cfg_vec)

    preds = surrogate.predict(np.array(rows, dtype=float))
    scored = sorted(zip(preds.tolist(), candidates), key=lambda x: x[0], reverse=True)
    return scored


def _formula_fallback(meta_features: Dict[str, float]) -> Dict[str, Any]:
    """Formula-based sizing when pkl files are absent (always available)."""
    n_feat = max(1, int(meta_features.get("num_features", 16)))
    n_cls = max(2, int(meta_features.get("num_classes", 2)))
    n_samp = max(100, int(meta_features.get("num_samples", 1000)))

    # Geometric-mean first layer width (Heaton rule)
    first_dim = max(16, int(math.sqrt(n_feat * n_cls)) * 4)
    first_dim = min(first_dim, 512)

    # Depth: log2 heuristic clamped to [1, 4]
    n_layers = max(1, min(4, int(math.log2(max(2, n_samp / max(n_feat, 1))))))

    hidden_dims = [max(8, first_dim // (2 ** i)) for i in range(n_layers)]
    imbalance = meta_features.get("imbalance_ratio", 1.0)
    dropout = 0.4 if (n_samp < 1000 or imbalance > 5) else 0.2

    return {
        "num_layers": n_layers,
        "max_units": first_dim,
        "hidden_dims": hidden_dims,
        "learning_rate": 1e-3,
        "dropout": dropout,
        "batch_size": 32 if n_samp < 5000 else 64,
        "source": "formula_fallback",
    }


def recommend_ann_config(meta_features: Dict[str, float]) -> Dict[str, Any]:
    """
    Main entry point called by model_selector.
    Returns a config dict with keys: hidden_dims, learning_rate, dropout,
    batch_size, num_layers, source.
    """
    result = _load_artifacts()
    if result is None:
        # Surrogate not available — use formula fallback
        cfg = _formula_fallback(meta_features)
        return cfg

    # Stage 1: k-NN warm-start
    candidates = knn_candidates(meta_features, k=3)
    if not candidates:
        return _formula_fallback(meta_features)

    # Stage 2: Surrogate scoring — pick best
    scored = score_configs(candidates, meta_features)
    if not scored:
        return _formula_fallback(meta_features)

    _, best_cfg = scored[0]

    num_layers = int(best_cfg.get("num_layers", 2))
    max_units = int(best_cfg.get("max_units", 128))

    return {
        "num_layers": num_layers,
        "max_units": max_units,
        "hidden_dims": _config_to_hidden_dims(num_layers, max_units),
        "learning_rate": float(best_cfg.get("learning_rate", 1e-3)),
        "dropout": float(best_cfg.get("dropout", 0.3)),
        "batch_size": int(best_cfg.get("batch_size", 64)),
        "source": "lcbench_surrogate",
    }
