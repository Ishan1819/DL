"""
test_surrogate.py
=================
Quick verification script. Run after train_surrogate.py:

    python scripts/test_surrogate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.surrogate_selector import (
    knn_candidates,
    recommend_ann_config,
    score_configs,
    _load_artifacts,
)


def test_pkl_files_exist():
    meta_pkl = ROOT / "data" / "lcbench_meta.pkl"
    surrogate_pkl = ROOT / "data" / "lcbench_surrogate.pkl"
    assert meta_pkl.exists(), f"FAIL: {meta_pkl} not found. Run train_surrogate.py first."
    assert surrogate_pkl.exists(), f"FAIL: {surrogate_pkl} not found."
    print("[PASS] pkl files exist")


def test_artifacts_load():
    result = _load_artifacts()
    assert result is not None, "FAIL: _load_artifacts() returned None"
    meta_store, surrogate = result
    assert len(meta_store) > 0, "FAIL: meta_store is empty"
    print(f"[PASS] artifacts loaded — {len(meta_store)} datasets in meta_store")


def test_knn_candidates():
    meta_features = {
        "num_samples": 5000.0,
        "num_features": 20.0,
        "num_classes": 3.0,
        "imbalance_ratio": 2.0,
        "missing_ratio": 0.0,
    }
    candidates = knn_candidates(meta_features, k=3)
    assert len(candidates) == 3, f"FAIL: expected 3 candidates, got {len(candidates)}"
    for cfg in candidates:
        assert "num_layers" in cfg, "FAIL: 'num_layers' missing from candidate"
        assert "learning_rate" in cfg, "FAIL: 'learning_rate' missing from candidate"
    print(f"[PASS] knn_candidates returned {len(candidates)} configs")


def test_score_configs():
    meta_features = {
        "num_samples": 5000.0,
        "num_features": 20.0,
        "num_classes": 3.0,
        "imbalance_ratio": 2.0,
        "missing_ratio": 0.0,
    }
    candidates = knn_candidates(meta_features, k=3)
    scored = score_configs(candidates, meta_features)
    assert len(scored) == 3, "FAIL: scored list length mismatch"
    # Should be sorted descending
    scores = [s for s, _ in scored]
    assert scores == sorted(scores, reverse=True), "FAIL: scored not sorted descending"
    print(f"[PASS] score_configs — best predicted val_acc: {scores[0]:.4f}")


def test_recommend_ann_config():
    meta_features = {
        "num_samples": 5000.0,
        "num_features": 20.0,
        "num_classes": 3.0,
        "imbalance_ratio": 2.0,
        "missing_ratio": 0.0,
    }
    cfg = recommend_ann_config(meta_features)
    assert "hidden_dims" in cfg, "FAIL: 'hidden_dims' missing"
    assert "learning_rate" in cfg, "FAIL: 'learning_rate' missing"
    assert "dropout" in cfg, "FAIL: 'dropout' missing"
    assert "batch_size" in cfg, "FAIL: 'batch_size' missing"
    assert isinstance(cfg["hidden_dims"], list) and len(cfg["hidden_dims"]) >= 1
    assert 1e-6 <= cfg["learning_rate"] <= 0.5, f"FAIL: lr={cfg['learning_rate']} out of range"
    assert 1 <= cfg["num_layers"] <= 5, f"FAIL: num_layers={cfg['num_layers']} out of range"
    print(f"[PASS] recommend_ann_config → hidden_dims={cfg['hidden_dims']}, "
          f"lr={cfg['learning_rate']}, dropout={cfg['dropout']}, "
          f"batch_size={cfg['batch_size']}, source={cfg.get('source')}")


def test_formula_fallback():
    """Verify fallback works even without pkl files (by using tiny dataset meta)."""
    from src.surrogate_selector import _formula_fallback
    cfg = _formula_fallback({"num_samples": 200.0, "num_features": 5.0, "num_classes": 2.0})
    assert "hidden_dims" in cfg
    assert "source" in cfg and cfg["source"] == "formula_fallback"
    print(f"[PASS] formula_fallback → {cfg['hidden_dims']}")


if __name__ == "__main__":
    print("=" * 55)
    print("LCBench Surrogate — Test Suite")
    print("=" * 55)
    test_pkl_files_exist()
    test_artifacts_load()
    test_knn_candidates()
    test_score_configs()
    test_recommend_ann_config()
    test_formula_fallback()
    print("=" * 55)
    print("All tests passed!")
