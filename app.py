from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data_handler import load_data, preprocess_data, split_data, validate_data
from src.explainer import generate_xai_results
from src.llm_generator import ask_llm_question, generate_explanation, suggest_config_with_llm
from src.meta_extractor import extract_meta_features
from src.model_selector import build_tabular_model, generate_configs, recommend_model
from src.task_detector import TaskInfo, get_task_info
from src.trainer import evaluate_model, select_best_model, train_models
from src.utils import ensure_dir, load_config, save_confusion_matrix, save_text_report_pdf, save_training_curves, setup_logging


setup_logging()
BASE_DIR = Path(__file__).resolve().parent
CONFIG = load_config(BASE_DIR / "config.yaml")

TABULAR_PREP = CONFIG.get("preprocessing", {}).get("tabular", {})
TAB_ASSUME_PREPROCESSED = bool(TABULAR_PREP.get("assume_preprocessed", False))
TAB_ENCODING_STRATEGY = str(TABULAR_PREP.get("encoding_strategy", "label")).lower()
TAB_REMOVE_OUTLIERS = bool(TABULAR_PREP.get("remove_outliers", True))
TAB_OUTLIER_IQR_K = float(TABULAR_PREP.get("outlier_iqr_k", 1.5))

UPLOAD_DIR = ensure_dir(BASE_DIR / CONFIG["paths"]["upload_dir"])
MODELS_DIR = ensure_dir(BASE_DIR / CONFIG["paths"]["models_dir"])
PLOTS_DIR = ensure_dir(BASE_DIR / CONFIG["paths"]["plots_dir"])


st.set_page_config(page_title="Intelligent Deep Learning Model Recommender", layout="wide")
st.title("Intelligent Deep Learning Model Recommender")


def _init_state() -> None:
    defaults = {
        "raw_data": None,
        "data_type": None,
        "data_summary": None,
        "task_info": None,
        "meta_features": None,
        "configs": None,
        "trained_models": None,
        "best_model_result": None,
        "test_metrics": None,
        "xai_results": None,
        "llm_report": None,
        "selected_configs": None,
        "target_col": None,
        "train_data": None,
        "val_data": None,
        "test_data": None,
        "processed_data_cache": None,
        "processed_data_signature": None,
        "llm_chat_history": [],
        "llm_suggestion": None,
        "uploaded_file_token": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


with st.sidebar:
    st.header("Controls")
    if st.button("Clear session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.session_state.get("trained_models"):
        st.subheader("Model Comparison")
        rows: List[Dict[str, Any]] = []
        for idx, tm in enumerate(st.session_state["trained_models"], start=1):
            rows.append(
                {
                    "Config": idx,
                    "Val Metric": tm.val_metric,
                    "Val Loss": tm.val_loss,
                    "LR": tm.config.get("learning_rate"),
                    "Batch": tm.config.get("batch_size"),
                    "Optimizer": tm.config.get("optimizer"),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")

    best = st.session_state.get("best_model_result")
    if best:
        with open(best.model_path, "rb") as f:
            st.download_button("Download best model", f.read(), file_name=Path(best.model_path).name, mime="application/octet-stream")


st.header("Step 1: Data Upload")
uploaded = st.file_uploader("Upload CSV, ZIP (images), TXT, or JSONL", type=["csv", "zip", "txt", "jsonl"])

if uploaded is not None:
    upload_token = (uploaded.name, int(getattr(uploaded, "size", 0)))
    if st.session_state.get("uploaded_file_token") != upload_token:
        save_path = UPLOAD_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Loading and validating data..."):
            try:
                raw_data, data_type = load_data(save_path)
                data_summary = validate_data(raw_data)
                st.session_state["raw_data"] = raw_data
                st.session_state["data_type"] = data_type
                st.session_state["data_summary"] = data_summary
                st.session_state["uploaded_file_token"] = upload_token

                # Reset downstream caches/state because dataset changed.
                st.session_state["processed_data_cache"] = None
                st.session_state["processed_data_signature"] = None
                st.session_state["train_data"] = None
                st.session_state["val_data"] = None
                st.session_state["test_data"] = None
                st.session_state["trained_models"] = None
                st.session_state["best_model_result"] = None

                st.success(f"Loaded {uploaded.name} as {data_type} data")
            except Exception as e:  # pylint: disable=broad-except
                st.error(f"Data loading failed: {e}")

if st.session_state.get("raw_data") is not None:
    data_type = st.session_state["data_type"]
    st.subheader("Data Preview")
    if data_type == "tabular":
        df = st.session_state["raw_data"]
        st.dataframe(df.head(5), width="stretch")
        st.write("Shape:", df.shape)
        st.write("Missing values:", pd.Series(st.session_state["data_summary"]["missing_values"]))
        st.caption(
            "Tabular preprocessing rules: "
            f"assume_preprocessed={TAB_ASSUME_PREPROCESSED}, "
            f"encoding={TAB_ENCODING_STRATEGY}, "
            f"remove_outliers={TAB_REMOVE_OUTLIERS}, "
            f"iqr_k={TAB_OUTLIER_IQR_K}"
        )
        target_col = st.selectbox("Select target column", options=df.columns.tolist(), index=len(df.columns) - 1)
        st.session_state["target_col"] = target_col
    elif data_type == "image":
        img_data = st.session_state["raw_data"]
        st.write(f"Number of images: {len(img_data.get('images', []))}")
        sample_images = img_data.get("images", [])[:3]
        if sample_images:
            cols = st.columns(len(sample_images))
            for i, image in enumerate(sample_images):
                cols[i].image(image, caption=f"Sample {i+1}", width="stretch")
    elif data_type == "text":
        texts = st.session_state["raw_data"]
        st.write(f"Number of text samples: {len(texts)}")
        st.write("Samples:")
        for t in texts[:5]:
            st.code(t)


st.header("Step 2: Task Detection")
if st.session_state.get("raw_data") is not None:
    with st.spinner("Detecting task type..."):
        try:
            task_info: TaskInfo = get_task_info(st.session_state["raw_data"], st.session_state.get("target_col"))
            st.session_state["task_info"] = task_info

            st.info(
                f"Data Type: {task_info.data_type} | Task: {task_info.task_type} | "
                f"Classes: {task_info.num_classes} | Features: {task_info.num_features}"
            )

            if task_info.data_type == "tabular" and st.session_state.get("target_col"):
                meta = extract_meta_features(st.session_state["raw_data"], st.session_state["target_col"])
                st.session_state["meta_features"] = meta
                st.subheader("Meta-features")
                st.json({k: round(v, 4) for k, v in list(meta.items())[:25]})
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Task detection failed: {e}")


st.header("Step 3: Model Recommendation")
if st.session_state.get("task_info"):
    task_info = st.session_state["task_info"]
    recs = recommend_model(task_info, st.session_state.get("meta_features"))
    configs = generate_configs(task_info, st.session_state.get("meta_features"))
    st.session_state["configs"] = configs

    for rec in recs:
        st.markdown(f"**{rec.model_name}**")
        st.write(rec.notes)
        st.json(rec.architecture)

    selected = st.multiselect(
        "Select configs to train",
        options=[c["config_id"] for c in configs],
        default=[c["config_id"] for c in configs],
    )
    st.session_state["selected_configs"] = selected

    with st.expander("Hyperparameter configurations"):
        for c in configs:
            st.json(c)


st.header("Step 4: Training")
if st.session_state.get("task_info") and st.button("Start Training", type="primary"):
    task_info = st.session_state["task_info"]

    if task_info.data_type != "tabular":
        st.warning("Current training pipeline in this release is optimized for tabular workflows. Recommendation details for image/text are still provided.")
    else:
        try:
            prep_signature = (
                tuple(st.session_state["raw_data"].columns.tolist()) if isinstance(st.session_state.get("raw_data"), pd.DataFrame) else None,
                tuple(st.session_state["raw_data"].shape) if isinstance(st.session_state.get("raw_data"), pd.DataFrame) else None,
                st.session_state.get("target_col"),
                TAB_ASSUME_PREPROCESSED,
                TAB_ENCODING_STRATEGY,
                TAB_REMOVE_OUTLIERS,
                TAB_OUTLIER_IQR_K,
            )

            prep_text = "Validating preprocessed data and splitting..." if TAB_ASSUME_PREPROCESSED else "Preprocessing and splitting data..."
            with st.spinner(prep_text):
                if (
                    st.session_state.get("processed_data_cache") is not None
                    and st.session_state.get("processed_data_signature") == prep_signature
                ):
                    processed = st.session_state["processed_data_cache"]
                    st.info("Reusing cached cleaned dataset (no re-preprocessing).")
                else:
                    processed = preprocess_data(
                        st.session_state["raw_data"],
                        "tabular",
                        st.session_state["target_col"],
                        tabular_encoding=TAB_ENCODING_STRATEGY,
                        remove_outliers=TAB_REMOVE_OUTLIERS,
                        outlier_iqr_k=TAB_OUTLIER_IQR_K,
                        assume_preprocessed=TAB_ASSUME_PREPROCESSED,
                    )
                    st.session_state["processed_data_cache"] = processed
                    st.session_state["processed_data_signature"] = prep_signature

                cleaned_df = processed.get("cleaned_data_df")
                if cleaned_df is not None:
                    with st.expander("Preview cleaned data"):
                        st.write({"processing_mode": processed.get("processing_mode", "unknown")})
                        st.dataframe(cleaned_df.head(20), width="stretch")
                        st.write({"rows": int(cleaned_df.shape[0]), "columns": int(cleaned_df.shape[1])})
                        st.download_button(
                            "Download cleaned data (CSV)",
                            data=cleaned_df.to_csv(index=False),
                            file_name="cleaned_data.csv",
                            mime="text/csv",
                        )

                train_data, val_data, test_data = split_data(processed, stratify="classification" in task_info.task_type)
                st.session_state["train_data"] = train_data
                st.session_state["val_data"] = val_data
                st.session_state["test_data"] = test_data

            selected_configs = [
                c
                for c in st.session_state["configs"]
                if c["config_id"] in (st.session_state.get("selected_configs") or [])
            ]

            if not selected_configs:
                st.error("Please select at least one configuration")
            else:
                progress = st.progress(0.0)
                metrics_placeholder = st.empty()

                def _progress_callback(stage: str, fraction: float, metric_map: Dict[str, float]) -> None:
                    progress.progress(min(max(fraction, 0.0), 1.0))
                    metrics_placeholder.write(
                        {
                            "stage": stage,
                            "epoch": int(metric_map.get("epoch", 0)),
                            "train_loss": round(metric_map.get("train_loss", 0.0), 5),
                            "val_loss": round(metric_map.get("val_loss", 0.0), 5),
                            "val_metric": round(metric_map.get("val_metric", 0.0), 5),
                        }
                    )

                with st.spinner("Training models with Optuna + LLM warm-start..."):
                    trained_models = train_models(
                        task_info=task_info,
                        configs=selected_configs,
                        train_data=train_data,
                        val_data=val_data,
                        models_dir=MODELS_DIR,
                        num_optuna_trials=int(CONFIG["training"]["num_optuna_trials"]),
                        early_stopping_patience=int(CONFIG["training"]["early_stopping_patience"]),
                        max_epochs=int(CONFIG["training"]["max_epochs"]),
                        progress_callback=_progress_callback,
                        warm_start_config=None,
                        # LLM cross-config warm-start: each config's trial 0 uses LLM suggestion
                        llm_provider=CONFIG["api"].get("use_llm", "groq"),
                        llm_api_keys=CONFIG["api"],
                        meta_features=st.session_state.get("meta_features"),
                    )

                if not trained_models:
                    st.error("All training configurations failed. Check logs and data quality.")
                else:
                    best = select_best_model(trained_models, task_info)
                    st.session_state["trained_models"] = trained_models
                    st.session_state["best_model_result"] = best
                    st.session_state["llm_chat_history"] = []  # reset chat on new training
                    st.success("Training completed successfully")

                    # ── LLM Config Suggestion (runs after training) ──────────
                    with st.spinner("Asking LLM for a smarter next config..."):
                        prior = [
                            {"config": tm.config, "val_metric": tm.val_metric}
                            for tm in trained_models
                        ]
                        suggestion = suggest_config_with_llm(
                            previous_results=prior,
                            meta_features=st.session_state.get("meta_features"),
                            llm_provider=CONFIG["api"].get("use_llm", "gemini"),
                            api_keys=CONFIG["api"],
                        )
                        st.session_state["llm_suggestion"] = suggestion

        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Training failed: {e}")


st.header("Step 5: Results")
if st.session_state.get("best_model_result") and st.session_state.get("task_info"):
    task_info = st.session_state["task_info"]
    best = st.session_state["best_model_result"]

    trained_models = st.session_state.get("trained_models") or []
    if trained_models:
        st.subheader("Training Summary")
        summary_rows: List[Dict[str, Any]] = []
        for idx, tm in enumerate(trained_models, start=1):
            summary_rows.append(
                {
                    "Config": int(tm.config.get("config_id", idx)),
                    "Best Trial Objective": round(float(tm.config.get("best_trial_objective", float("nan"))), 5)
                    if tm.config.get("best_trial_objective") is not None
                    else None,
                    "Mapped Val Accuracy": round(float(tm.config.get("mapped_val_accuracy", float("nan"))), 5)
                    if tm.config.get("mapped_val_accuracy") is not None
                    else None,
                    "Final Retrain Val Accuracy": round(float(tm.config.get("final_retrain_val_accuracy", tm.val_metric)), 5)
                    if tm.config.get("final_retrain_val_accuracy") is not None
                    else round(float(tm.val_metric), 5),
                    "Adaptive Retry Used": bool(tm.config.get("adaptive_retry", False)),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")

    model = build_tabular_model(task_info, best.config)
    state = torch.load(best.model_path, map_location="cpu")
    model.load_state_dict(state)

    metrics = evaluate_model(model, st.session_state["test_data"], task_info)
    st.session_state["test_metrics"] = metrics

    st.subheader("Best Model")
    st.write({"model": best.model_name, "config": best.config, "val_metric": best.val_metric, "val_loss": best.val_loss})

    curves_path = save_training_curves(
        best.history,
        PLOTS_DIR / "training_curves.png",
        metric_name="accuracy" if "classification" in task_info.task_type else "fitness",
    )
    st.image(str(curves_path), caption="Training Curves", width="stretch")

    if metrics.get("confusion_matrix") is not None:
        label_names = [str(v) for v in metrics.get("labels", [])]
        cm_path = save_confusion_matrix(metrics["confusion_matrix"], label_names, PLOTS_DIR / "confusion_matrix.png")
        st.image(str(cm_path), caption="Confusion Matrix", width="stretch")

    st.subheader("Test Metrics")
    st.json({k: v for k, v in metrics.items() if k not in {"confusion_matrix", "predictions", "actual", "labels"}})


st.header("Step 6: Explainability")
if st.session_state.get("best_model_result") and st.session_state.get("train_data") is not None:
    if st.button("Generate SHAP & LIME"):
        try:
            task_info = st.session_state["task_info"]
            if task_info.data_type != "tabular" or "regression" in task_info.task_type:
                st.warning("Current explainability pipeline in this release supports tabular classification best.")
            else:
                model = build_tabular_model(task_info, st.session_state["best_model_result"].config)
                model.load_state_dict(torch.load(st.session_state["best_model_result"].model_path, map_location="cpu"))

                with st.spinner("Generating XAI artifacts..."):
                    xai_results = generate_xai_results(
                        model=model,
                        X_train=np.array(st.session_state["train_data"]["X"].toarray() if hasattr(st.session_state["train_data"]["X"], "toarray") else st.session_state["train_data"]["X"]),
                        X_test=np.array(st.session_state["test_data"]["X"].toarray() if hasattr(st.session_state["test_data"]["X"], "toarray") else st.session_state["test_data"]["X"]),
                        feature_names=st.session_state["train_data"].get("feature_names", []),
                        output_dir=PLOTS_DIR,
                    )
                st.session_state["xai_results"] = xai_results

                shap_paths = xai_results.get("shap", {}).get("plot_paths", {})
                if shap_paths.get("summary") and os.path.exists(shap_paths["summary"]):
                    st.image(shap_paths["summary"], caption="SHAP Summary", width="stretch")
                    with open(shap_paths["summary"], "rb") as f:
                        st.download_button("Download SHAP summary", f.read(), "shap_summary.png", "image/png")

                if shap_paths.get("waterfall") and os.path.exists(shap_paths["waterfall"]):
                    st.image(shap_paths["waterfall"], caption="SHAP Waterfall", width="stretch")
                    with open(shap_paths["waterfall"], "rb") as f:
                        st.download_button("Download SHAP waterfall", f.read(), "shap_waterfall.png", "image/png")

                lime_html = xai_results.get("lime", {}).get("html_path", "")
                if lime_html and os.path.exists(lime_html):
                    with open(lime_html, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.download_button("Download LIME HTML", html_content, "lime_explanation.html", "text/html")

        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Explainability failed: {e}")


# ── Best Config Panel (shown after training) ─────────────────────────────────
if st.session_state.get("best_model_result"):
    best_result = st.session_state["best_model_result"]
    with st.expander("🏆 Best Config Found", expanded=True):
        clean_cfg = {
            k: v for k, v in best_result.config.items()
            if k not in {"input_dim", "config_id", "best_trial_objective", "mapped_val_accuracy",
                         "mapped_val_metric", "final_retrain_val_accuracy", "final_retrain_val_metric",
                         "adaptive_retry", "hidden_profile"}
        }
        st.json(clean_cfg)
        st.caption(f"Val accuracy: **{round(best_result.val_metric, 4)}** | Val loss: **{round(best_result.val_loss, 4)}**")


st.header("Step 7: AI-Generated Insights")
if st.session_state.get("xai_results") and st.session_state.get("test_metrics"):
    if st.button("Generate AI Report"):
        with st.spinner("Calling LLM API..."):
            report = generate_explanation(
                xai_results=st.session_state["xai_results"],
                model_info={
                    "model_type": st.session_state["best_model_result"].model_name if st.session_state.get("best_model_result") else "Unknown",
                    "task_type": st.session_state["task_info"].task_type if st.session_state.get("task_info") else "Unknown",
                    "train_acc": (
                        st.session_state["best_model_result"].history.train_metric[-1]
                        if st.session_state.get("best_model_result")
                        else "N/A"
                    ),
                },
                metrics=st.session_state["test_metrics"],
                llm_provider=CONFIG["api"].get("use_llm", "gemini"),
                api_keys=CONFIG["api"],
            )
            st.session_state["llm_report"] = report

if st.session_state.get("llm_report"):
    st.markdown(st.session_state["llm_report"])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = save_text_report_pdf(st.session_state["llm_report"], tmp.name)
    with open(pdf_path, "rb") as f:
        st.download_button("Download report as PDF", f.read(), file_name="model_insights_report.pdf", mime="application/pdf")

    # ── Human-in-the-loop Chat Q&A ──────────────────────────────────────────
    st.subheader("💬 Ask the AI About Your Model")
    st.caption(
        "Ask any question about the model, its accuracy, SHAP/LIME explanations, or how to improve it. "
        "The AI has full context of your training results."
    )

    # Replay existing conversation history
    for msg in st.session_state.get("llm_chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    user_q = st.chat_input("e.g. Which feature is most important? Why is accuracy low?")
    if user_q:
        # Store and display user message
        st.session_state["llm_chat_history"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Build context and get LLM answer
        best_result = st.session_state.get("best_model_result")
        chat_context = {
            "model_info": {
                "model_type": best_result.model_name if best_result else "Unknown",
                "task_type": st.session_state["task_info"].task_type if st.session_state.get("task_info") else "Unknown",
                "train_acc": (
                    best_result.history.train_metric[-1] if best_result else "N/A"
                ),
            },
            "test_metrics": st.session_state.get("test_metrics", {}),
            "xai_results": st.session_state.get("xai_results", {}),
            "best_config": best_result.config if best_result else {},
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_llm_question(
                    question=user_q,
                    context=chat_context,
                    llm_provider=CONFIG["api"].get("use_llm", "gemini"),
                    api_keys=CONFIG["api"],
                )
            st.markdown(answer)

        st.session_state["llm_chat_history"].append({"role": "assistant", "content": answer})
