from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional

import anthropic

try:
    import groq as groq_sdk
except Exception:  # pylint: disable=broad-except
    groq_sdk = None

try:
    from google import genai
except Exception:  # pylint: disable=broad-except
    genai = None

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.generativeai")
    try:
        import google.generativeai as legacy_genai
    except Exception:  # pylint: disable=broad-except
        legacy_genai = None

from .utils import get_logger


LOGGER = get_logger(__name__)


PROMPT_TEMPLATE = """You are an AI model explainer. Based on the following information, generate a comprehensive natural language summary report:

Model Type: {model_type}
Task: {task_type}
Training Accuracy: {train_acc}
Test Accuracy: {test_acc}
Top Important Features: {features}
SHAP Analysis: {shap_summary}
LIME Analysis: {lime_summary}

Generate a report with:
1. Model Performance Summary
2. Key Insights from Feature Importance
3. Recommendations for Model Improvement
4. Limitations and Considerations
"""


def _build_prompt(xai_results: Dict[str, Any], model_info: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    shap_top_features: List[str] = xai_results.get("shap", {}).get("top_features", [])
    shap_summary = f"Top features: {shap_top_features}" if shap_top_features else "SHAP summary unavailable"

    lime_summary_items = xai_results.get("lime", {}).get("summary", [])
    lime_summary = ", ".join([f"{k}: {v:.4f}" for k, v in lime_summary_items[:5]]) if lime_summary_items else "LIME summary unavailable"

    return PROMPT_TEMPLATE.format(
        model_type=model_info.get("model_type", "Unknown"),
        task_type=model_info.get("task_type", "Unknown"),
        train_acc=model_info.get("train_acc", "N/A"),
        test_acc=metrics.get("accuracy", metrics.get("rmse", "N/A")),
        features=", ".join(shap_top_features[:5]) if shap_top_features else "N/A",
        shap_summary=shap_summary,
        lime_summary=lime_summary,
    )


def generate_explanation(
    xai_results: Dict[str, Any],
    model_info: Dict[str, Any],
    metrics: Dict[str, Any],
    llm_provider: str,
    api_keys: Dict[str, str],
) -> str:
    """Generate natural-language explanation using Groq, Gemini, or Claude API."""
    prompt = _build_prompt(xai_results, model_info, metrics)

    # \u2500\u2500 Groq (primary) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if llm_provider.lower() == "groq":
        api_key = api_keys.get("groq_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return "Groq API key not configured. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)
        if groq_sdk is None:
            return "groq package not installed. Run: pip install groq\n\n" + _fallback_report(model_info, metrics, xai_results)
        try:
            client = groq_sdk.Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )
            text = completion.choices[0].message.content
            return text or _fallback_report(model_info, metrics, xai_results)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Groq API failed: %s", e)
            return "Groq call failed. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

    # \u2500\u2500 Gemini \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if llm_provider.lower() == "gemini":
        api_key = api_keys.get("gemini_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return "Gemini API key not configured. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)
        try:
            if genai is not None:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                response_text = getattr(response, "text", None)
                if response_text:
                    return response_text
            if legacy_genai is not None:
                legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text or _fallback_report(model_info, metrics, xai_results)
            return _fallback_report(model_info, metrics, xai_results)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Gemini API failed: %s", e)
            return "LLM call failed. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

    # \u2500\u2500 Claude \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if llm_provider.lower() == "claude":
        api_key = api_keys.get("claude_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return "Claude API key not configured. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            if message.content and hasattr(message.content[0], "text"):
                return message.content[0].text
            return _fallback_report(model_info, metrics, xai_results)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Claude API failed: %s", e)
            return "LLM call failed. Showing fallback report.\n\n" + _fallback_report(model_info, metrics, xai_results)

    return _fallback_report(model_info, metrics, xai_results)


def _fallback_report(model_info: Dict[str, Any], metrics: Dict[str, Any], xai_results: Dict[str, Any]) -> str:
    """Create fallback explanation when external API is unavailable."""
    top_features = xai_results.get("shap", {}).get("top_features", [])
    lines = [
        "## 1. Model Performance Summary",
        f"- Model Type: {model_info.get('model_type', 'Unknown')}",
        f"- Task: {model_info.get('task_type', 'Unknown')}",
        f"- Test Accuracy/Fitness: {metrics.get('accuracy', metrics.get('rmse', 'N/A'))}",
        "",
        "## 2. Key Insights from Feature Importance",
        f"- Top SHAP features: {', '.join(top_features[:5]) if top_features else 'Unavailable'}",
        "",
        "## 3. Recommendations for Model Improvement",
        "- Tune learning rate, batch size, and hidden dimensions further.",
        "- Consider more training data and feature engineering.",
        "",
        "## 4. Limitations and Considerations",
        "- Explanation quality depends on model stability and feature quality.",
        "- LIME local explanations may vary per instance.",
    ]
    return "\n".join(lines)


# ── Helper: call any configured LLM with a raw prompt ────────────────────────

def _call_llm(prompt: str, llm_provider: str, api_keys: Dict[str, str]) -> str:
    """Dispatch a raw prompt to Gemini, Claude, or Groq and return the text response."""
    if llm_provider.lower() == "groq":
        api_key = api_keys.get("groq_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return ""
        if groq_sdk is None:
            LOGGER.warning("groq package not installed. Run: pip install groq")
            return ""
        try:
            client = groq_sdk.Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Groq API failed: %s", e)
        return ""

    if llm_provider.lower() == "gemini":
        api_key = api_keys.get("gemini_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return ""
        try:
            if genai is not None:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                text = getattr(response, "text", None)
                if text:
                    return text
            if legacy_genai is not None:
                legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text or ""
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Gemini API failed: %s", e)
        return ""

    if llm_provider.lower() == "claude":
        api_key = api_keys.get("claude_api_key", "")
        if not api_key or "YOUR_" in api_key:
            return ""
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            if message.content and hasattr(message.content[0], "text"):
                return message.content[0].text
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Claude API failed: %s", e)
        return ""

    return ""


# ── Human-in-the-loop Q&A ────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are an expert AI assistant helping a data scientist understand their trained deep learning model.
You have access to the following context about the model and its results. Answer the user's question clearly and concisely,
referencing specific numbers or features from the context where possible.

CONTEXT
-------
Model Type: {model_type}
Task: {task_type}
Train Accuracy (last epoch): {train_acc}
Test Accuracy: {test_acc}
Test Metrics: {test_metrics}
Top SHAP Features: {shap_features}
SHAP Summary: {shap_summary}
LIME Summary: {lime_summary}
Best Config: {best_config}

USER QUESTION: {question}
"""


def ask_llm_question(
    question: str,
    context: Dict[str, Any],
    llm_provider: str,
    api_keys: Dict[str, str],
) -> str:
    """Answer a free-text user question using the LLM with full model context.

    Parameters
    ----------
    question : str
        The user's free-text question.
    context : dict
        Keys expected: model_info, test_metrics, xai_results, best_config.
    llm_provider : str
        'gemini' or 'claude'.
    api_keys : dict
        API key mapping.

    Returns
    -------
    str
        LLM answer, or a fallback string if the API is unavailable.
    """
    model_info  = context.get("model_info", {})
    test_metrics = context.get("test_metrics", {})
    xai_results  = context.get("xai_results", {})
    best_config  = context.get("best_config", {})

    shap_top = xai_results.get("shap", {}).get("top_features", [])
    shap_summary = f"Top features: {shap_top[:8]}" if shap_top else "SHAP unavailable"
    lime_items = xai_results.get("lime", {}).get("summary", [])
    lime_summary = (
        ", ".join(f"{k}: {v:.4f}" for k, v in lime_items[:5]) if lime_items else "LIME unavailable"
    )
    display_metrics = {k: v for k, v in test_metrics.items() if k not in {"confusion_matrix", "predictions", "actual", "labels"}}

    prompt = QA_SYSTEM_PROMPT.format(
        model_type=model_info.get("model_type", "Unknown"),
        task_type=model_info.get("task_type", "Unknown"),
        train_acc=model_info.get("train_acc", "N/A"),
        test_acc=test_metrics.get("accuracy", test_metrics.get("rmse", "N/A")),
        test_metrics=json.dumps(display_metrics, default=str),
        shap_features=", ".join(shap_top[:8]) if shap_top else "N/A",
        shap_summary=shap_summary,
        lime_summary=lime_summary,
        best_config=json.dumps(best_config, default=str),
        question=question,
    )

    answer = _call_llm(prompt, llm_provider, api_keys)
    if not answer:
        return (
            f"*LLM unavailable — check your API key in config.yaml.*\n\n"
            f"**Your question:** {question}\n\n"
            f"**Quick context:** Model={model_info.get('model_type','?')}, "
            f"Test Acc={test_metrics.get('accuracy','N/A')}, "
            f"Top SHAP features: {', '.join(shap_top[:3]) if shap_top else 'N/A'}"
        )
    return answer


# ── LLM-based hyperparameter suggestion (BO alternative) ─────────────────────

CONFIG_SUGGEST_PROMPT = """You are an expert ML hyperparameter optimizer.
You have seen the following training configurations and their validation accuracies.
Your job is to suggest ONE new configuration that you predict will achieve HIGHER validation accuracy.

PREVIOUS RESULTS (sorted worst to best)
----------------------------------------
{history}

DATASET META-FEATURES
----------------------
{meta_features}

Based on the trends above, suggest a new configuration. Reply with ONLY a JSON object (no markdown fences) containing these keys:
  learning_rate  (float, e.g. 0.001)
  batch_size     (int, one of: 16, 32, 64, 128, 256)
  optimizer      ("Adam" or "SGD")
  dropout        (float between 0.0 and 0.7)
  hidden_dims    (list of ints, e.g. [256, 128, 64])
  reason         (string: one sentence explaining your choice)
"""


def suggest_config_with_llm(
    previous_results: List[Dict[str, Any]],
    meta_features: Optional[Dict[str, float]],
    llm_provider: str,
    api_keys: Dict[str, str],
) -> Dict[str, Any]:
    """Ask the LLM to suggest the next hyperparameter config based on prior results.

    Parameters
    ----------
    previous_results : list of dicts
        Each dict: {'config': {...}, 'val_metric': float}.
    meta_features : dict or None
        Dataset meta-features.
    llm_provider : str
        'gemini' or 'claude'.
    api_keys : dict
        API key mapping.

    Returns
    -------
    dict
        Suggested hyperparameters including a 'reason' key, or the best
        previous config as fallback.
    """
    if not previous_results:
        return {}

    sorted_results = sorted(previous_results, key=lambda r: r.get("val_metric", 0.0))
    history_lines = []
    for i, r in enumerate(sorted_results, 1):
        cfg = r.get("config", {})
        history_lines.append(
            f"{i}. val_accuracy={r.get('val_metric', 0):.4f} | "
            f"lr={cfg.get('learning_rate','?')} | "
            f"batch={cfg.get('batch_size','?')} | "
            f"opt={cfg.get('optimizer','?')} | "
            f"dropout={cfg.get('dropout','?')} | "
            f"hidden={cfg.get('hidden_dims','?')}"
        )
    history_str = "\n".join(history_lines)

    meta_str = (
        ", ".join(f"{k}={round(v, 4)}" for k, v in list((meta_features or {}).items())[:10])
        or "not available"
    )

    prompt = CONFIG_SUGGEST_PROMPT.format(history=history_str, meta_features=meta_str)
    raw = _call_llm(prompt, llm_provider, api_keys)

    if raw:
        try:
            # Strip any accidental markdown fences
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            suggestion = json.loads(cleaned)
            # Validate required keys
            required = {"learning_rate", "batch_size", "optimizer", "dropout", "hidden_dims"}
            if required.issubset(suggestion.keys()):
                return suggestion
        except (json.JSONDecodeError, AttributeError) as e:
            LOGGER.warning("LLM config suggestion parse failed: %s — raw: %s", e, raw[:200])

    # Fallback: return the best previous config
    best = sorted_results[-1].get("config", {})
    best["reason"] = "LLM suggestion unavailable. Returning the best observed config as fallback."
    return best
