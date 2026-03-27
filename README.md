# Intelligent Deep Learning Model Recommender

Production-ready Streamlit system to:

1. Upload tabular/image/text datasets.
2. Auto-detect task type.
3. Extract tabular meta-features (PyMFE).
4. Recommend deep learning architectures.
5. Train multiple configs with Optuna + early stopping.
6. Evaluate and select best model.
7. Generate SHAP + LIME explainability artifacts.
8. Generate natural-language findings with Gemini/Claude (or fallback report).

## Project Structure

- `data/uploads`: uploaded input files
- `data/models`: saved checkpoints and preprocessing artifacts
- `src/data_handler.py`: loading, validation, preprocessing, splitting
- `src/task_detector.py`: data/task detection + `TaskInfo`
- `src/meta_extractor.py`: PyMFE-based meta-feature extraction
- `src/model_selector.py`: architecture recommendations + config generation
- `src/trainer.py`: Optuna training, best-model selection, evaluation
- `src/explainer.py`: SHAP + LIME generation
- `src/llm_generator.py`: Gemini/Claude report generation
- `src/utils.py`: logging, plotting, utility helpers
- `app.py`: Streamlit UI workflow
- `config.yaml`: API/training/paths configuration
- `requirements.txt`: dependencies

## Python Version

- Python `3.10+`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Update API keys in `config.yaml`:
   - `api.claude_api_key`
   - `api.gemini_api_key`
   - `api.use_llm` (`gemini` or `claude`)

## Run

- `streamlit run app.py`

## Streamlit Workflow

1. **Step 1** Upload data and preview.
2. **Step 2** Detect data type/task and meta-features.
3. **Step 3** Review recommended models and configs.
4. **Step 4** Train selected configs with progress + live metrics.
5. **Step 5** View best model metrics and confusion matrix.
6. **Step 6** Generate SHAP/LIME explanations.
7. **Step 7** Generate LLM insights and download PDF report.

## Notes

- Tabular training/evaluation/XAI is fully implemented.
- Image and text recommendation paths are included and detectable in the UI.
- All long-running operations are wrapped with Streamlit progress/spinner feedback.
- Errors are handled with user-friendly messages and logging.

## Outputs Saved to Disk

- Trained model checkpoints: `data/models/*.pth`
- Preprocessors/encoders: `data/models/*.joblib`
- Curves/confusion/XAI plots: `data/models/plots/*`
- LIME HTML explanation: `data/models/plots/lime_explanation.html`
- Generated report PDF: downloaded from UI
