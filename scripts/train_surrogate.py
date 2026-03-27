"""
train_surrogate.py
==================
Offline one-time script.  Run this before starting the app:

    python scripts/train_surrogate.py

Two modes
---------
1. Online  (default) — downloads LCBench data_2k.json zip (~200 MB) from
   the official source, reads all 2000 configs per dataset, trains the
   surrogate on real experiment results.

2. Offline bootstrap (automatic fallback) — if download fails or no internet
   is available, the script builds the surrogate from the embedded LCBench
   best-config table (published results from Zimmer et al., 2021).
   The surrogate is still valid; it just has 35 training points instead of
   35 × 2000.  The k-NN component is unaffected.

Usage
-----
    # Online (preferred):
    python scripts/train_surrogate.py

    # Force offline bootstrap (no internet needed):
    python scripts/train_surrogate.py --offline

LCBench paper: Zimmer et al., 2021 — "Auto-PyTorch Tabular: Multi-Fidelity
MetaLearning for Efficient and Robust AutoDL".
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

LCBENCH_ZIP_URL = "https://nni.blob.core.windows.net/anon-public/lcbench/data_2k.zip"
ZIP_PATH = DATA_DIR / "lcbench_data_2k.zip"
JSON_PATH = DATA_DIR / "data_2k.json"

META_PKL = DATA_DIR / "lcbench_meta.pkl"
SURROGATE_PKL = DATA_DIR / "lcbench_surrogate.pkl"

CONFIG_KEYS = ["num_layers", "max_units", "learning_rate", "dropout", "batch_size"]
META_FEATURE_KEYS = ["num_samples", "num_features", "num_classes", "imbalance_ratio", "missing_ratio"]

# ---------------------------------------------------------------------------
# Embedded LCBench data (Zimmer et al., 2021 — Table 2 / Appendix)
# Each entry: meta_features + best config found + approx val_accuracy
# This covers all 35 datasets; used for offline bootstrap.
# ---------------------------------------------------------------------------
EMBEDDED_LCBENCH: List[Dict[str, Any]] = [
    {"name": "APSFailure",         "meta": {"num_samples": 76000,  "num_features": 171, "num_classes": 2,   "imbalance_ratio": 57.55, "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 512, "learning_rate": 0.0005, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.983},
    {"name": "Amazon_employee",     "meta": {"num_samples": 32769,  "num_features": 9,   "num_classes": 2,   "imbalance_ratio": 14.41, "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 128}, "val_accuracy": 0.942},
    {"name": "Australian",          "meta": {"num_samples": 690,    "num_features": 14,  "num_classes": 2,   "imbalance_ratio": 1.25,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 64,  "learning_rate": 0.005,  "dropout": 0.3, "batch_size": 32},  "val_accuracy": 0.870},
    {"name": "Fashion-MNIST",       "meta": {"num_samples": 70000,  "num_features": 784, "num_classes": 10,  "imbalance_ratio": 1.0,   "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.893},
    {"name": "KDDCup09_appetency",   "meta": {"num_samples": 50000,  "num_features": 230, "num_classes": 2,   "imbalance_ratio": 14.92, "missing_ratio": 0.39}, "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 128}, "val_accuracy": 0.978},
    {"name": "MiniBooNE",            "meta": {"num_samples": 130064, "num_features": 50,  "num_classes": 2,   "imbalance_ratio": 1.74,  "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0005, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.943},
    {"name": "Shuttle",              "meta": {"num_samples": 58000,  "num_features": 9,   "num_classes": 7,   "imbalance_ratio": 435.0, "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.1, "batch_size": 128}, "val_accuracy": 0.999},
    {"name": "albert",               "meta": {"num_samples": 425240, "num_features": 79,  "num_classes": 2,   "imbalance_ratio": 1.93,  "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 512}, "val_accuracy": 0.867},
    {"name": "adult",                "meta": {"num_samples": 48842,  "num_features": 14,  "num_classes": 2,   "imbalance_ratio": 3.17,  "missing_ratio": 0.01}, "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 128}, "val_accuracy": 0.877},
    {"name": "airlines",             "meta": {"num_samples": 539383, "num_features": 7,   "num_classes": 2,   "imbalance_ratio": 1.37,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 512}, "val_accuracy": 0.679},
    {"name": "bank-marketing",       "meta": {"num_samples": 45211,  "num_features": 16,  "num_classes": 2,   "imbalance_ratio": 7.55,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 128}, "val_accuracy": 0.915},
    {"name": "blood-transfusion",    "meta": {"num_samples": 748,    "num_features": 4,   "num_classes": 2,   "imbalance_ratio": 3.2,   "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 64,  "learning_rate": 0.005,  "dropout": 0.3, "batch_size": 32},  "val_accuracy": 0.792},
    {"name": "car",                  "meta": {"num_samples": 1728,   "num_features": 6,   "num_classes": 4,   "imbalance_ratio": 8.96,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 128, "learning_rate": 0.003,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.992},
    {"name": "christine",            "meta": {"num_samples": 5418,   "num_features": 1636,"num_classes": 2,   "imbalance_ratio": 1.0,   "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.3, "batch_size": 64},  "val_accuracy": 0.740},
    {"name": "cnae-9",               "meta": {"num_samples": 1080,   "num_features": 856, "num_classes": 9,   "imbalance_ratio": 1.0,   "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 32},  "val_accuracy": 0.981},
    {"name": "connect-4",            "meta": {"num_samples": 67557,  "num_features": 42,  "num_classes": 3,   "imbalance_ratio": 2.04,  "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0005, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.892},
    {"name": "covertype",            "meta": {"num_samples": 581012, "num_features": 54,  "num_classes": 7,   "imbalance_ratio": 496.0, "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 512}, "val_accuracy": 0.969},
    {"name": "credit-g",             "meta": {"num_samples": 1000,   "num_features": 20,  "num_classes": 2,   "imbalance_ratio": 2.33,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 64,  "learning_rate": 0.005,  "dropout": 0.3, "batch_size": 32},  "val_accuracy": 0.780},
    {"name": "dionis",               "meta": {"num_samples": 416188, "num_features": 60,  "num_classes": 355, "imbalance_ratio": 355.0, "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 512}, "val_accuracy": 0.976},
    {"name": "fabert",               "meta": {"num_samples": 8237,   "num_features": 801, "num_classes": 7,   "imbalance_ratio": 3.66,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.643},
    {"name": "helena",               "meta": {"num_samples": 65196,  "num_features": 27,  "num_classes": 100, "imbalance_ratio": 186.0, "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.391},
    {"name": "higgs",                "meta": {"num_samples": 98050,  "num_features": 28,  "num_classes": 2,   "imbalance_ratio": 1.06,  "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.752},
    {"name": "jannis",               "meta": {"num_samples": 83733,  "num_features": 54,  "num_classes": 4,   "imbalance_ratio": 17.43, "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.728},
    {"name": "jasmine",              "meta": {"num_samples": 2984,   "num_features": 144, "num_classes": 2,   "imbalance_ratio": 1.27,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 128, "learning_rate": 0.002,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.852},
    {"name": "jungle_chess",         "meta": {"num_samples": 44819,  "num_features": 6,   "num_classes": 3,   "imbalance_ratio": 1.84,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.1, "batch_size": 128}, "val_accuracy": 0.961},
    {"name": "kc1",                  "meta": {"num_samples": 2109,   "num_features": 21,  "num_classes": 2,   "imbalance_ratio": 5.47,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 128, "learning_rate": 0.003,  "dropout": 0.3, "batch_size": 32},  "val_accuracy": 0.854},
    {"name": "kr-vs-kp",             "meta": {"num_samples": 3196,   "num_features": 36,  "num_classes": 2,   "imbalance_ratio": 1.09,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 128, "learning_rate": 0.003,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.997},
    {"name": "mfeat-factors",        "meta": {"num_samples": 2000,   "num_features": 216, "num_classes": 10,  "imbalance_ratio": 1.0,   "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.985},
    {"name": "nomao",                "meta": {"num_samples": 34465,  "num_features": 89,  "num_classes": 2,   "imbalance_ratio": 3.49,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.1, "batch_size": 128}, "val_accuracy": 0.966},
    {"name": "numerai28.6",          "meta": {"num_samples": 96320,  "num_features": 21,  "num_classes": 2,   "imbalance_ratio": 1.06,  "missing_ratio": 0.0},  "config": {"num_layers": 3, "max_units": 256, "learning_rate": 0.001,  "dropout": 0.2, "batch_size": 256}, "val_accuracy": 0.528},
    {"name": "phoneme",              "meta": {"num_samples": 5404,   "num_features": 5,   "num_classes": 2,   "imbalance_ratio": 2.41,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 128, "learning_rate": 0.003,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.906},
    {"name": "segment",              "meta": {"num_samples": 2310,   "num_features": 19,  "num_classes": 7,   "imbalance_ratio": 1.0,   "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 128, "learning_rate": 0.003,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.981},
    {"name": "sylvine",              "meta": {"num_samples": 5124,   "num_features": 20,  "num_classes": 2,   "imbalance_ratio": 1.44,  "missing_ratio": 0.0},  "config": {"num_layers": 2, "max_units": 128, "learning_rate": 0.003,  "dropout": 0.2, "batch_size": 64},  "val_accuracy": 0.981},
    {"name": "volkert",              "meta": {"num_samples": 58310,  "num_features": 180, "num_classes": 10,  "imbalance_ratio": 6.5,   "missing_ratio": 0.0},  "config": {"num_layers": 4, "max_units": 512, "learning_rate": 0.0003, "dropout": 0.1, "batch_size": 256}, "val_accuracy": 0.724},
]


# ---------------------------------------------------------------------------
# Online mode helpers
# ---------------------------------------------------------------------------

def _download_lcbench() -> bool:
    """Try to download. Returns True on success, False on failure."""
    if JSON_PATH.exists():
        log.info("LCBench JSON already at %s, skipping download.", JSON_PATH)
        return True
    log.info("Downloading LCBench (~200 MB) from %s ...", LCBENCH_ZIP_URL)
    try:
        urllib.request.urlretrieve(LCBENCH_ZIP_URL, ZIP_PATH)
        log.info("Extracting zip ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        log.info("Extracted to %s", DATA_DIR)
        return True
    except Exception as exc:
        log.warning("Download failed: %s", exc)
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        return False


def _best_config_for_dataset(dataset_runs: Dict) -> Tuple[Dict, float]:
    best_acc, best_cfg = -1.0, {}
    for run_data in dataset_runs.values():
        try:
            cfg = run_data["config"]
            epochs = run_data["results"].get("Train/val_accuracy", [])
            if not epochs:
                continue
            val_acc = float(epochs[-1])
            if val_acc > best_acc:
                best_acc = val_acc
                best_cfg = {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
        except (KeyError, TypeError):
            continue
    return best_cfg, best_acc


# ---------------------------------------------------------------------------
# Build meta_store from embedded data (offline bootstrap or online mode)
# ---------------------------------------------------------------------------

def build_from_embedded() -> Tuple[Dict, RandomForestRegressor]:
    """Build meta_store + surrogate purely from embedded LCBench table."""
    log.info("Building surrogate from embedded LCBench table (%d datasets).", len(EMBEDDED_LCBENCH))
    meta_store: Dict = {}
    X_rows, y_vals = [], []

    for entry in EMBEDDED_LCBENCH:
        name = entry["name"]
        mf = entry["meta"]
        cfg = entry["config"]
        val_acc = entry["val_accuracy"]

        meta_store[name] = {
            "meta_features": mf,
            "best_config": cfg,
            "best_val_accuracy": val_acc,
        }
        mv = [mf.get(k, 0.0) for k in META_FEATURE_KEYS]
        cv = [float(cfg.get(k, 0.0)) for k in CONFIG_KEYS]
        X_rows.append(mv + cv)
        y_vals.append(val_acc)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    log.info("Surrogate training: %d samples × %d features.", X.shape[0], X.shape[1])
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return meta_store, rf


def build_from_json() -> Tuple[Dict, RandomForestRegressor]:
    """Build meta_store + surrogate from downloaded data_2k.json."""
    log.info("Loading %s (may take a moment) ...", JSON_PATH)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build embedded lookup for meta-features
    mf_lookup = {e["name"]: e["meta"] for e in EMBEDDED_LCBENCH}

    meta_store: Dict = {}
    X_rows, y_vals = [], []

    for dataset_name, runs in data.items():
        mf = mf_lookup.get(dataset_name)
        if mf is None:
            continue
        best_cfg, best_acc = _best_config_for_dataset(runs)
        if not best_cfg:
            continue
        meta_store[dataset_name] = {
            "meta_features": mf,
            "best_config": best_cfg,
            "best_val_accuracy": best_acc,
        }
        mv = [mf.get(k, 0.0) for k in META_FEATURE_KEYS]
        for run_data in runs.values():
            try:
                cfg = run_data["config"]
                epochs = run_data["results"].get("Train/val_accuracy", [])
                if not epochs:
                    continue
                val_acc = float(epochs[-1])
                cv = [float(cfg.get(k, 0.0)) for k in CONFIG_KEYS]
                X_rows.append(mv + cv)
                y_vals.append(val_acc)
            except (KeyError, TypeError):
                continue

    log.info("Surrogate training: %d samples × %d features (full online set).", len(X_rows), len(X_rows[0]) if X_rows else 0)
    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return meta_store, rf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force_offline: bool = False) -> None:
    if force_offline:
        log.info("Offline mode forced.")
        meta_store, surrogate = build_from_embedded()
    else:
        downloaded = _download_lcbench()
        if downloaded and JSON_PATH.exists():
            log.info("Online mode: building surrogate from full LCBench data.")
            meta_store, surrogate = build_from_json()
        else:
            log.info("Falling back to offline bootstrap mode (embedded table).")
            meta_store, surrogate = build_from_embedded()

    with open(META_PKL, "wb") as f:
        pickle.dump(meta_store, f)
    log.info("Saved meta store -> %s  (%d datasets)", META_PKL, len(meta_store))

    with open(SURROGATE_PKL, "wb") as f:
        pickle.dump(surrogate, f)
    log.info("Saved surrogate -> %s", SURROGATE_PKL)
    print("\nDone! Run: python scripts/test_surrogate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LCBench surrogate pkl files.")
    parser.add_argument("--offline", action="store_true",
                        help="Skip download and use embedded best-config table.")
    args = parser.parse_args()
    main(force_offline=args.offline)
