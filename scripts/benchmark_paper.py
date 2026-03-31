import time
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# ML metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Baselines
from flaml import AutoML
import optuna

# PyTorch (for the Optuna DL baseline)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# OpenML for dataset fetching
import openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure you have installed: pip install openml scikit-learn pandas numpy flaml optuna torch
# NOTE: Set up your time budget (in seconds)
TIME_BUDGET = 60  


class SimpleMLP(nn.Module):
    """A standard MLP architecture for tabular data."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_optuna_dl(X_train, y_train, X_val, y_val, time_budget, num_classes):
    """Deep Learning Baseline with Optuna hyperparameter tuning."""
    start_time = time.time()
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=128, shuffle=False)

    def objective(trial):
        if time.time() - start_time > time_budget:
            raise optuna.exceptions.TrialPruned("Out of time budget.")
            
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
        model = SimpleMLP(X_train.shape[1], hidden_dim, num_classes, num_layers, dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a maximal set of epochs, but break early if out of time
        best_val_acc = 0.0
        for epoch in range(10):
            if time.time() - start_time > time_budget:
                break
                
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = model(bx)
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == by).sum().item()
                    total += by.size(0)
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
        return best_val_acc

    # Suppress optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    try:
        # We optimize using the exact same wall-clock time budget as FLAML
        study.optimize(objective, timeout=time_budget)
    except Exception as e:
        pass
        
    # In a real paper, you'd retrain the best DL model and predict on X_test here.
    # For this script we assume validation accuracy as the proxy benchmark.
    return study.best_value if len(study.trials) > 0 else 0.0


def extract_meta_features(X, y):
    """
    Extract simple dataset meta-features to figure out *when* algorithms fail.
    In a real paper, use library `pymfe`.
    """
    return {
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "num_classes": len(np.unique(y)),
        "class_imbalance_ratio": float(np.max(np.bincount(y)) / np.min(np.bincount(y)))
    }


def main():
    # Let's use the OpenML CC-18 benchmarking suite (a standard for ML papers)
    # Suite ID 99 is standard OpenML-CC18 classification benchmark
    suite = openml.study.get_suite(99)
    dataset_ids = suite.data[:10]  # Just doing 10 for demonstration; use suite.data for all ~72
    
    results = []
    
    for task_idx, did in enumerate(dataset_ids):
        logging.info(f"Processing Dataset {task_idx+1}/{len(dataset_ids)}: ID {did}")
        try:
            dataset = openml.datasets.get_dataset(did, download_data=True)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            
            # Simple Preprocessing
            X = pd.get_dummies(X, drop_first=True)
            X = SimpleImputer(strategy="mean").fit_transform(X)
            X = StandardScaler().fit_transform(X)
            y = LabelEncoder().fit_transform(y)
            num_classes = len(np.unique(y))
            
            # Extract Meta-Features
            meta_features = extract_meta_features(X, y)
            
            # Standard Split: Train 60%, Val 20%, Test 20%
            X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)
            
            row = {"Dataset_ID": did, **meta_features}
            
            # ====================================================
            # 1. Baseline: Untuned Random Forest (Sanity Check)
            # ====================================================
            rf_start = time.time()
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_time = time.time() - rf_start
            
            y_pred_rf = rf.predict(X_test)
            row["RF_Acc"] = accuracy_score(y_test, y_pred_rf)
            row["RF_Time"] = rf_time
            
            # ====================================================
            # 2. Baseline: FLAML AutoML
            # ====================================================
            automl = AutoML()
            settings = {
                "time_budget": TIME_BUDGET,
                "metric": 'accuracy',
                "task": 'classification',
                "log_file_name": "flaml.log",
                "verbose": 0
            }
            
            flaml_start = time.time()
            automl.fit(X_train=X_train, y_train=y_train, **settings)
            flaml_time = time.time() - flaml_start
            
            y_pred_flaml = automl.predict(X_test)
            row["FLAML_Acc"] = accuracy_score(y_test, y_pred_flaml)
            row["FLAML_Time"] = flaml_time
            row["FLAML_Best_Estimator"] = automl.best_estimator
            
            # ====================================================
            # 3. Baseline: Optuna Tuned Deep Learning
            # ====================================================
            optuna_start = time.time()
            # For strictness, the DL model searches using X_train and evaluates on X_val
            dl_val_acc = train_optuna_dl(X_train, y_train, X_val, y_val, TIME_BUDGET, num_classes)
            optuna_time = time.time() - optuna_start
            
            row["Optuna_DL_Val_Acc"] = dl_val_acc
            row["Optuna_DL_Time"] = optuna_time
            
            results.append(row)
            logging.info(f"Dataset {did} Completed: RF={row['RF_Acc']:.3f}, FLAML={row['FLAML_Acc']:.3f}, DL={row['Optuna_DL_Val_Acc']:.3f}")
            
        except Exception as e:
            logging.error(f"Failed on Dataset {did}: {e}")
            
    # Save Benchmark Results to CSV for Paper Analysis
    out_df = pd.DataFrame(results)
    out_df.to_csv("benchmark_paper_results.csv", index=False)
    logging.info("Benchmarking completed! Results saved to 'benchmark_paper_results.csv'.")
    print("\n--- Summary of Wins ---")
    if not out_df.empty:
        print(out_df[["Dataset_ID", "num_samples", "RF_Acc", "FLAML_Acc", "Optuna_DL_Val_Acc"]].to_string())

if __name__ == "__main__":
    main()
