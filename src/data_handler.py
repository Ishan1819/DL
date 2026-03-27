from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from transformers import AutoTokenizer

from .utils import get_logger


LOGGER = get_logger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TEXT_EXTENSIONS = {".txt", ".jsonl"}


def _prepare_preprocessed_tabular(data: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Use user-provided cleaned tabular data as-is (no feature preprocessing)."""
    X = data.drop(columns=[target]).copy()
    y_raw = data[target].copy()

    if X.isna().any().any():
        raise ValueError("Dataset marked preprocessed but still contains missing values")

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Dataset marked preprocessed but non-numeric feature columns found: {non_numeric[:5]}"
        )

    feature_names = X.columns.tolist()
    X_processed = X.to_numpy(dtype=np.float32)

    y_encoder = None
    y = y_raw.copy()
    if y.dtype == "object" or y.nunique(dropna=True) <= max(20, int(len(y) * 0.05)):
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y_raw.astype(str))

    cleaned_features_df = X.copy()
    cleaned_data_df = data.copy()

    return {
        "X": X_processed,
        "y": np.array(y),
        "preprocessor": None,
        "target_encoder": y_encoder,
        "feature_names": feature_names,
        "cleaned_features_df": cleaned_features_df,
        "cleaned_data_df": cleaned_data_df,
        "processing_mode": "preprocessed_input",
    }


def load_data(file_path: str | Path) -> tuple[Any, str]:
    """Auto-detect and load CSV, image folder/zip, or text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), "tabular"

    if path.suffix.lower() == ".zip":
        extract_dir = path.parent / f"{path.stem}_unzipped"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        return _load_image_folder(extract_dir), "image"

    if path.is_dir():
        return _load_image_folder(path), "image"

    if path.suffix.lower() in TEXT_EXTENSIONS:
        if path.suffix.lower() == ".jsonl":
            texts = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    texts.append(obj.get("text", ""))
            return texts, "text"
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()], "text"

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return _load_image_folder(path.parent), "image"

    raise ValueError("Unsupported file format. Upload CSV, image folder/zip, TXT, or JSONL.")


def _load_image_folder(folder: Path) -> Dict[str, Any]:
    """Load images and inferred labels from folder structure."""
    images: List[np.ndarray] = []
    labels: List[str] = []
    paths: List[str] = []

    image_paths = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    for p in image_paths:
        image = cv2.imread(str(p))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        labels.append(p.parent.name)
        paths.append(str(p))

    return {"images": images, "labels": labels, "paths": paths, "bounding_boxes": None}


def validate_data(data: Any) -> Dict[str, Any]:
    """Validate data quality and return summary stats."""
    summary: Dict[str, Any] = {}

    if isinstance(data, pd.DataFrame):
        summary = {
            "shape": data.shape,
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isna().sum().to_dict(),
            "describe": data.describe(include="all").fillna("N/A").to_dict(),
        }
    elif isinstance(data, dict) and "images" in data:
        summary = {
            "num_images": len(data.get("images", [])),
            "num_classes": len(set(data.get("labels", []))) if data.get("labels") else 0,
            "sample_paths": data.get("paths", [])[:5],
        }
    elif isinstance(data, list):
        summary = {
            "num_records": len(data),
            "avg_text_length": float(np.mean([len(x) for x in data])) if data else 0,
            "sample_texts": data[:3],
        }
    else:
        raise ValueError("Unsupported data format for validation")

    return summary


def preprocess_data(
    data: Any,
    data_type: str,
    target: Optional[str] = None,
    *,
    tabular_encoding: str = "label",
    remove_outliers: bool = True,
    outlier_iqr_k: float = 1.5,
    assume_preprocessed: bool = False,
) -> Dict[str, Any]:
    """Preprocess tabular/image/text data according to type."""
    if data_type == "tabular":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Tabular preprocessing expects pandas DataFrame")
        if target is None or target not in data.columns:
            raise ValueError("Valid target column is required for tabular processing")

        if assume_preprocessed:
            return _prepare_preprocessed_tabular(data, target)

        X = data.drop(columns=[target]).copy()
        y_raw = data[target].copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if remove_outliers and numeric_cols:
            mask = pd.Series(True, index=X.index)
            for col in numeric_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or not np.isfinite(iqr):
                    continue
                lower = q1 - outlier_iqr_k * iqr
                upper = q3 + outlier_iqr_k * iqr
                mask &= X[col].between(lower, upper) | X[col].isna()
            if mask.sum() > 0:
                X = X.loc[mask].reset_index(drop=True)
                y_raw = y_raw.loc[mask].reset_index(drop=True)
        if tabular_encoding == "onehot":
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ]
            )

            X_processed = preprocessor.fit_transform(X)
            feature_names = _extract_feature_names(preprocessor, numeric_cols, categorical_cols)

        else:
            X_clean = X.copy()
            preprocessor = {
                "encoding": "label",
                "category_label_maps": {},
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            }

            for col in numeric_cols:
                fill_value = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(fill_value)

            for col in categorical_cols:
                if X_clean[col].isna().all():
                    X_clean[col] = "missing"
                else:
                    mode = X_clean[col].mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else "missing"
                    X_clean[col] = X_clean[col].fillna(fill_value)

                le = LabelEncoder()
                X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                preprocessor["category_label_maps"][col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

            if numeric_cols:
                scaler = StandardScaler()
                X_clean[numeric_cols] = scaler.fit_transform(X_clean[numeric_cols])
                preprocessor["scaler_mean_"] = scaler.mean_.tolist()
                preprocessor["scaler_scale_"] = scaler.scale_.tolist()

            feature_names = X_clean.columns.tolist()
            X_processed = X_clean.to_numpy(dtype=np.float32)

        y_encoder = None
        y = y_raw.copy()
        if y.dtype == "object" or y.nunique(dropna=True) <= max(20, int(len(y) * 0.05)):
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y_raw.astype(str))

        if hasattr(X_processed, "toarray"):
            cleaned_features_df = pd.DataFrame.sparse.from_spmatrix(X_processed, columns=feature_names)
        else:
            cleaned_features_df = pd.DataFrame(X_processed, columns=feature_names)

        cleaned_data_df = cleaned_features_df.copy()
        cleaned_data_df[target] = y_raw.values

        return {
            "X": X_processed,
            "y": np.array(y),
            "preprocessor": preprocessor,
            "target_encoder": y_encoder,
            "feature_names": feature_names,
            "cleaned_features_df": cleaned_features_df,
            "cleaned_data_df": cleaned_data_df,
            "processing_mode": f"pipeline_{tabular_encoding}",
        }

    if data_type == "image":
        if not isinstance(data, dict) or "images" not in data:
            raise TypeError("Image preprocessing expects dict with `images`")

        resized = []
        for img in data["images"]:
            img_resized = cv2.resize(img, (224, 224))
            resized.append(img_resized.astype(np.float32) / 255.0)

        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(data.get("labels", [])) if data.get("labels") else np.array([])
        return {
            "X": np.array(resized),
            "y": y,
            "target_encoder": y_encoder,
            "feature_names": ["pixel_values"],
        }

    if data_type == "text":
        texts = data if isinstance(data, list) else []
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        return {
            "X": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "y": np.zeros(len(texts), dtype=np.int64),
            "tokenizer": tokenizer,
            "feature_names": ["tokens"],
        }

    raise ValueError(f"Unsupported data type: {data_type}")


def split_data(
    processed_data: Dict[str, Any],
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split processed data into 70/15/15 train/val/test sets."""
    X = processed_data["X"]
    y = processed_data["y"]

    def _can_stratify(y_arr: Any) -> bool:
        y_np = np.array(y_arr)
        if y_np.size == 0:
            return False
        if y_np.dtype.kind not in {"i", "u"}:
            return False
        values, counts = np.unique(y_np, return_counts=True)
        return len(values) > 1 and int(np.min(counts)) >= 2

    stratify_y = y if stratify and _can_stratify(y) else None
    if stratify and stratify_y is None:
        LOGGER.warning("Stratified split disabled: class distribution does not meet minimum count >= 2 per class.")

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=random_state,
            stratify=stratify_y,
        )
    except ValueError:
        LOGGER.warning("Primary stratified split failed. Falling back to non-stratified split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=random_state,
            stratify=None,
        )

    stratify_temp = y_temp if stratify and _can_stratify(y_temp) else None
    if stratify and stratify_temp is None:
        LOGGER.warning("Validation/test stratified split disabled due to small class counts in temporary split.")

    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=random_state,
            stratify=stratify_temp,
        )
    except ValueError:
        LOGGER.warning("Validation/test stratified split failed. Falling back to non-stratified split.")
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=random_state,
            stratify=None,
        )

    common = {k: v for k, v in processed_data.items() if k not in {"X", "y"}}

    train_data = {"X": X_train, "y": y_train, **common}
    val_data = {"X": X_val, "y": y_val, **common}
    test_data = {"X": X_test, "y": y_test, **common}
    return train_data, val_data, test_data


def _extract_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Extract transformed feature names from a fitted preprocessor."""
    feature_names = list(numeric_cols)

    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"]["encoder"]
        cat_names = encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_names)

    return feature_names
